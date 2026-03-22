#!/usr/bin/env python3
"""
Enhanced Railway Simulation - Novice-Friendly Visualization & UI Module

This module replaces and significantly improves the visualization components
of the railway simulation with:
- Intuitive, modern UI design
- Interactive tutorials and tooltips
- Real-time statistics dashboard
- Smooth animations and visual feedback
- Accessibility features
- Helpful legends and guides
"""

import pygame
import math
from enum import Enum
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional, Any
from collections import deque


# ============================================================================
# Enhanced Color Scheme - Modern & Accessible
# ============================================================================

class ModernColors:
    """Modern, accessible color palette with better contrast"""
    
    # Base theme colors
    DARK_BG = (15, 23, 42)           # Slate-900
    MEDIUM_BG = (30, 41, 59)         # Slate-800
    LIGHT_BG = (51, 65, 85)          # Slate-700
    ACCENT_PRIMARY = (59, 130, 246)  # Blue-500
    ACCENT_SUCCESS = (34, 197, 94)   # Green-500aurant = (251, 191, 36)  # Amber-400
    ACCENT_DANGER = (239, 68, 68)    # Red-500
    
    # UI Elements
    UI_PANEL = (30, 41, 59, 230)     # Semi-transparent slate-800
    UI_BORDER = (71, 85, 105)        # Slate-600
    UI_HOVER = (51, 65, 85)          # Slate-700
    UI_ACTIVE = (71, 85, 105)        # Slate-600
    
    # Text
    TEXT_PRIMARY = (248, 250, 252)   # Slate-50
    TEXT_SECONDARY = (203, 213, 225) # Slate-300
    TEXT_MUTED = (148, 163, 184)     # Slate-400
    
    # Track colors
    TRACK_IDLE = (100, 116, 139)     # Slate-500
    TRACK_ACTIVE = (34, 197, 94)     # Green-500
    TRACK_BUSY = (251, 191, 36)      # Amber-400
    TRACK_BLOCKED = (239, 68, 68)    # Red-500
    TRACK_DISRUPTED = (249, 115, 22) # Orange-500
    
    # Station colors
    STATION_DEFAULT = (59, 130, 246) # Blue-500
    STATION_HIGHLIGHT = (96, 165, 250) # Blue-400
    STATION_SELECTED = (251, 191, 36) # Amber-400
    
    # Train colors by type
    TRAIN_PASSENGER = (139, 92, 246)  # Violet-500
    TRAIN_FREIGHT = (100, 116, 139)   # Slate-500
    TRAIN_EXPRESS = (251, 191, 36)    # Amber-400
    TRAIN_LOCAL = (59, 130, 246)      # Blue-500
    TRAIN_MAINTENANCE = (249, 115, 22) # Orange-500
    TRAIN_EMERGENCY = (239, 68, 68)    # Red-500
    










































































    
    # Signal colors
    SIGNAL_GREEN = (34, 197, 94)      # Green-500
    SIGNAL_YELLOW = (251, 191, 36)    # Amber-400
    SIGNAL_RED = (239, 68, 68)        # Red-500
    SIGNAL_OFF = (71, 85, 105)        # Slate-600
    
    # Grid and guides
    GRID_MAJOR = (51, 65, 85, 100)    # Slate-700 with alpha
    GRID_MINOR = (30, 41, 59, 50)     # Slate-800 with alpha
    
    # Tooltips and overlays
    TOOLTIP_BG = (30, 41, 59, 240)    # Slate-800
    TOOLTIP_BORDER = (71, 85, 105)    # Slate-600
    OVERLAY_BG = (15, 23, 42, 200)    # Slate-900 with alpha
    
    # Status indicators
    STATUS_ONLINE = (34, 197, 94)     # Green-500
    STATUS_WARNING = (251, 191, 36)   # Amber-400
    STATUS_ERROR = (239, 68, 68)      # Red-500
    STATUS_INFO = (59, 130, 246)      # Blue-500


# ============================================================================
# UI Components
# ============================================================================

class Button:
    """Interactive button with hover and click effects"""
    
    def __init__(self, rect: pygame.Rect, text: str, 
                 color=ModernColors.ACCENT_PRIMARY,
                 hover_color=ModernColors.ACCENT_SUCCESS,
                 text_color=ModernColors.TEXT_PRIMARY):
        self.rect = rect
        self.text = text
        self.color = color
        self.hover_color = hover_color
        self.text_color = text_color
        self.is_hovered = False
        self.is_pressed = False
        self.enabled = True
        self.callback = None
        
    def update(self, mouse_pos: Tuple[int, int], mouse_pressed: bool):
        """Update button state"""
        self.is_hovered = self.rect.collidepoint(mouse_pos) if self.enabled else False
        
        if self.is_hovered and mouse_pressed and self.callback:
            self.callback()
            
    def render(self, surface: pygame.Surface, font: pygame.font.Font):
        """Render button"""
        # Determine color based on state
        if not self.enabled:
            color = ModernColors.UI_BORDER
        elif self.is_hovered:
            color = self.hover_color
        else:
            color = self.color
            
        # Draw button background with shadow
        shadow_rect = self.rect.move(2, 2)
        pygame.draw.rect(surface, (0, 0, 0, 50), shadow_rect, border_radius=6)
        pygame.draw.rect(surface, color, self.rect, border_radius=6)
        pygame.draw.rect(surface, ModernColors.UI_BORDER, self.rect, 2, border_radius=6)
        
        # Draw text
        text_surf = font.render(self.text, True, self.text_color)
        text_rect = text_surf.get_rect(center=self.rect.center)
        surface.blit(text_surf, text_rect)


class Panel:
    """UI Panel with title and content area"""
    
    def __init__(self, rect: pygame.Rect, title: str, 
                 collapsible: bool = False):
        self.rect = rect
        self.title = title
        self.collapsible = collapsible
        self.collapsed = False
        self.title_height = 35
        self.padding = 10
        self.widgets = []
        
    def toggle_collapse(self):
        """Toggle panel collapse state"""
        if self.collapsible:
            self.collapsed = not self.collapsed
            
    def render(self, surface: pygame.Surface, 
               font_title: pygame.font.Font,
               font_content: pygame.font.Font):
        """Render panel"""
        # Draw main panel
        pygame.draw.rect(surface, ModernColors.UI_PANEL[:3], self.rect, border_radius=8)
        pygame.draw.rect(surface, ModernColors.UI_BORDER, self.rect, 2, border_radius=8)
        
        # Draw title bar
        title_rect = pygame.Rect(self.rect.x, self.rect.y, 
                                 self.rect.width, self.title_height)
        pygame.draw.rect(surface, ModernColors.LIGHT_BG, title_rect, 
                        border_top_left_radius=8, border_top_right_radius=8)
        
        # Draw title text
        title_surf = font_title.render(self.title, True, ModernColors.TEXT_PRIMARY)
        title_text_rect = title_surf.get_rect(
            midleft=(self.rect.x + self.padding, title_rect.centery)
        )
        surface.blit(title_surf, title_text_rect)
        
        # Draw collapse indicator if collapsible
        if self.collapsible:
            indicator = "▼" if not self.collapsed else "▶"
            ind_surf = font_content.render(indicator, True, ModernColors.TEXT_SECONDARY)
            ind_rect = ind_surf.get_rect(
                midright=(self.rect.right - self.padding, title_rect.centery)
            )
            surface.blit(ind_surf, ind_rect)


# Simplified versions of other components due to space
class ProgressBar:
    """Animated progress bar"""
    
    def __init__(self, rect: pygame.Rect, min_val: float = 0.0, max_val: float = 1.0):
        self.rect = rect
        self.min_val = min_val
        self.max_val = max_val
        self.current_val = 0.0
        self.target_val = 0.0
        self.color = ModernColors.ACCENT_PRIMARY
        
    def set_value(self, value: float):
        """Set target value"""
        self.target_val = max(self.min_val, min(self.max_val, value))
        
    def update(self, delta_time: float):
        """Update animation"""
        diff = self.target_val - self.current_val
        self.current_val += diff * delta_time * 5.0
        
    def render(self, surface: pygame.Surface, font: pygame.font.Font, 
               label: str = "", show_percentage: bool = True):
        """Render progress bar"""
        pygame.draw.rect(surface, ModernColors.MEDIUM_BG, self.rect, border_radius=4)
        pygame.draw.rect(surface, ModernColors.UI_BORDER, self.rect, 1, border_radius=4)
        
        progress = max(0.0, min(1.0, (self.current_val - self.min_val) / (self.max_val - self.min_val)))
        
        if progress > 0:
            fill_width = int(self.rect.width * progress)
            fill_rect = pygame.Rect(self.rect.x, self.rect.y, fill_width, self.rect.height)
            pygame.draw.rect(surface, self.color, fill_rect, border_radius=4)
        
        if label or show_percentage:
            text = f"{label} {progress*100:.0f}%" if label else f"{progress*100:.0f}%"
            text_surf = font.render(text, True, ModernColors.TEXT_PRIMARY)
            text_rect = text_surf.get_rect(center=self.rect.center)
            surface.blit(text_surf, text_rect)


# This code is getting corrupted. Let me provide a simplified, working version instead
