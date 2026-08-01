from manim import *
import numpy as np

# Use the provided TeachingScene base class without modification.
class TeachingScene(Scene):
    def setup_layout(self, title_text, lecture_lines):
        # BASE
        self.camera.background_color = "#000000"
        self.title = Text(title_text, font_size=28, color=WHITE).to_edge(UP)
        self.add(self.title)

        # Left-side lecture content (bullets with "-")
        lecture_texts = [Text(line, font_size=22, color=WHITE) for line in lecture_lines]
        self.lecture = VGroup(*lecture_texts).arrange(DOWN, aligned_edge=LEFT).scale(0.8)
        self.lecture.to_edge(LEFT, buff=0.2)
        self.add(self.lecture)

        # Define fine-grained animation grid (4x4 grid on right side)
        self.grid = {}
        rows = ["A", "B", "C", "D", "E", "F"]  # Top to bottom
        cols = ["1", "2", "3", "4", "5", "6"]  # Left to right

        for i, row in enumerate(rows):
            for j, col in enumerate(cols):
                x = 0.5 + j * 1
                y = 2.2 - i * 1
                self.grid[f"{row}{col}"] = np.array([x, y, 0])

    def place_at_grid(self, mobject, grid_pos, scale_factor=1.0):
        mobject.scale(scale_factor)
        mobject.move_to(self.grid[grid_pos])
        return mobject

    def place_in_area(self, mobject, top_left, bottom_right, scale_factor=1.0):
        tl_pos = self.grid[top_left]
        br_pos = self.grid[bottom_right]
        
        # Calculate center of the area
        center_x = (tl_pos[0] + br_pos[0]) / 2
        center_y = (tl_pos[1] + br_pos[1]) / 2
        center = np.array([center_x, center_y, 0])
        
        mobject.scale(scale_factor)
        mobject.move_to(center)
        return mobject

class Section3Scene(TeachingScene):
    def construct(self):
        # Configuration
        TITLE_TEXT = "The Evidence: Conditional Likelihood"
        LECTURE_LINES = [
            "The detective uses a Spark Test on a bot.",
            "Ninety percent of Glitch-Bots spark when tested.",
            "But ten percent of Normal-Bots also spark.",
            "We slice the rectangles to show these likelihoods.",
            "Shaded areas represent the chance of seeing a spark."
        ]
        
        # Color definitions
        RED_BASE = "#E74C3C"      # Prior Red from Section 2
        GREEN_BASE = "#2ECC71"    # Prior Green from Section 2
        RED_SPARK = "#FF5555"     # Brighter Red for Likelihood
        GREEN_SPARK = "#55FF55"   # Brighter Green for Likelihood
        DIMMED = "#111111"        # Darkened for non-sparking regions
        HIGHLIGHT_COLOR = BLUE_B
        
        self.setup_layout(TITLE_TEXT, LECTURE_LINES)

        # Asset loading (Issue 22)
        detect_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/detect.svg")
        # Place it at the top right of the workspace area
        self.place_at_grid(detect_icon, "A6", scale_factor=0.6)

        # Initialization of geometric components
        # Glitch-Bot Rectangle (Col 2)
        glitch_rect = Rectangle(width=1, height=4, fill_color=RED_BASE, fill_opacity=0.8, stroke_color=WHITE, stroke_width=2)
        self.place_in_area(glitch_rect, "B2", "E2")
        
        # Normal-Bot Rectangle (Cols 3-5)
        normal_rect = Rectangle(width=3, height=4, fill_color=GREEN_BASE, fill_opacity=0.8, stroke_color=WHITE, stroke_width=2)
        self.place_in_area(normal_rect, "B3", "E5")
        
        # Base labels
        glitch_label = Text("Glitch-Bots", font_size=20, color=RED_BASE)
        self.place_at_grid(glitch_label, "F2", scale_factor=0.8)
        
        normal_label = Text("Normal-Bots", font_size=20, color=GREEN_BASE)
        # Fix Issue 32: Center 'Normal-Bots' label across the width of the rectangle
        self.place_in_area(normal_label, 'F3', 'F5', scale_factor=0.8)

        # === Animation for Lecture Line 1 ===
        # "The detective uses a Spark Test on a bot."
        self.lecture[0].set_color(YELLOW)
        self.play(
            FadeIn(glitch_rect),
            FadeIn(normal_rect),
            Write(glitch_label),
            Write(normal_label)
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # "Ninety percent of Glitch-Bots spark when tested."
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(RED_SPARK)
        
        # Top 90% Spark area for Red: Height 4 * 0.9 = 3.6
        glitch_spark = Rectangle(
            width=1, height=3.6, 
            fill_color=RED_SPARK, fill_opacity=0.9, 
            stroke_width=0
        )
        glitch_spark.move_to(glitch_rect.get_critical_point(UP), aligned_edge=UP)
        
        self.play(FadeIn(glitch_spark, shift=DOWN))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # "But ten percent of Normal-Bots also spark."
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(GREEN_SPARK)
        
        # Top 10% Spark area for Green: Height 4 * 0.1 = 0.4
        normal_spark = Rectangle(
            width=3, height=0.4,
            fill_color=GREEN_SPARK, fill_opacity=0.9,
            stroke_width=0
        )
        normal_spark.move_to(normal_rect.get_critical_point(UP), aligned_edge=UP)
        
        self.play(FadeIn(normal_spark, shift=DOWN))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # "We slice the rectangles to show these likelihoods."
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(HIGHLIGHT_COLOR)
        
        # Creating visual slicing lines at the boundary
        slice_line_red = Line(
            glitch_spark.get_critical_point(LEFT + DOWN),
            glitch_spark.get_critical_point(RIGHT + DOWN),
            color=WHITE, stroke_width=4
        )
        slice_line_normal = Line(
            normal_spark.get_critical_point(LEFT + DOWN),
            normal_spark.get_critical_point(RIGHT + DOWN),
            color=WHITE, stroke_width=4
        )
        
        # Pulse detect icon and highlight slice lines (Issue 22)
        self.play(
            Create(slice_line_red), 
            Create(slice_line_normal),
            FadeIn(detect_icon)
        )
        self.play(
            Indicate(slice_line_red, color=HIGHLIGHT_COLOR), 
            Indicate(slice_line_normal, color=HIGHLIGHT_COLOR),
            Indicate(detect_icon, color=HIGHLIGHT_COLOR),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # "Shaded areas represent the chance of seeing a spark."
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(ORANGE)
        
        # Dim the lower regions (No-Spark zones)
        glitch_dim = Rectangle(
            width=1, height=0.4,
            fill_color=DIMMED, fill_opacity=0.9, stroke_width=0
        ).move_to(glitch_rect.get_critical_point(DOWN), aligned_edge=DOWN)
        
        normal_dim = Rectangle(
            width=3, height=3.6,
            fill_color=DIMMED, fill_opacity=0.9, stroke_width=0
        ).move_to(normal_rect.get_critical_point(DOWN), aligned_edge=DOWN)
        
        # Specialized Likelihood Labels (True Positive vs False Positive)
        tp_label = Text("True Positive", font_size=18, color=WHITE)
        # Fix Issue 30: Use specific grid point and scale
        self.place_at_grid(tp_label, 'B2', scale_factor=0.6)
        
        fp_label = Text("False Positive", font_size=18, color=WHITE)
        # Fix Issue 31: Center 'False Positive' across the Normal-Bot rectangle's top area
        self.place_in_area(fp_label, 'B3', 'B5', scale_factor=0.6)

        self.play(
            FadeIn(glitch_dim),
            FadeIn(normal_dim),
            FadeIn(tp_label),
            FadeIn(fp_label)
        )
        self.wait(2)
