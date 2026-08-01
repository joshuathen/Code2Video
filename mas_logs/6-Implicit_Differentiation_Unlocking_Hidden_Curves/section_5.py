from manim import *
import numpy as np

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

class Section5Scene(TeachingScene):
    def construct(self):
        # Setup layout
        title_text = "Visual Validation: The Radar Cat"
        lecture_lines = [
            "At point three four, the slope is negative three-fourths.",
            "At the peak zero five, the slope is zero.",
            "Tangent lines perfectly match our calculated values."
        ]
        self.setup_layout(title_text, lecture_lines)

        # Colors
        COLOR_1 = "#FFFF66"  # Light Yellow
        COLOR_2 = "#66FFFF"  # Light Cyan
        COLOR_3 = "#66FF66"  # Light Green
        
        # Radar Cat character (stylized triangle + text)
        radar_cat = VGroup(
            Triangle(color=WHITE, fill_opacity=1).scale(0.12),
            Text("Radar Cat", font_size=14, color=WHITE)
        ).arrange(UP, buff=0.1)
        # Initial position at the center of the right area (grid D4)
        self.place_at_grid(radar_cat, "D4")

        # === Animation for Lecture Line 1 ===
        # Draw the yellow circle and highlight point (3, 4)
        self.play(self.lecture[0].animate.set_color(COLOR_1))
        
        # Circle: Center at D4, Radius 2.0 grid units (matches math radius 5)
        circle = Circle(radius=2.0, color=YELLOW)
        self.place_at_grid(circle, "D4")
        self.play(Create(circle))
        
        # Point (3, 4) calculation: (3/5)*2.0 = 1.2, (4/5)*2.0 = 1.6
        pos_3_4 = self.grid["D4"] + 1.2 * RIGHT + 1.6 * UP
        dot1 = Dot(pos_3_4, color=COLOR_1)
        # Replaced MathTex with Text to avoid LaTeX dependency errors
        label1 = Text("(3, 4)", font_size=20, color=WHITE).next_to(dot1, UR, buff=0.1)
        
        self.play(FadeIn(dot1), FadeIn(label1))
        self.play(FadeIn(radar_cat))
        self.play(radar_cat.animate.move_to(pos_3_4 + 0.4 * UR))
        
        # Show calculation text: 'dy/dx = -3/4'
        # Replaced MathTex with Text to avoid LaTeX dependency errors
        calc1 = Text("dy/dx = -3/4", font_size=24, color=COLOR_1)
        self.place_at_grid(calc1, "A6")
        self.play(Write(calc1))
        
        # Draw a tangent line through (3, 4) with slope -0.75
        tangent1 = Line(
            pos_3_4 + 1.0 * (LEFT + 0.75 * UP),
            pos_3_4 + 1.0 * (RIGHT + 0.75 * DOWN),
            color=COLOR_1
        )
        self.play(Create(tangent1))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Highlight point (0, 5) and show calculation 'dy/dx = -0/5 = 0'
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(COLOR_2)
        )
        
        # Point (0, 5) calculation: (0/5)*2.0 = 0, (5/5)*2.0 = 2.0 -> grid B4
        pos_0_5 = self.grid["B4"]
        dot2 = Dot(pos_0_5, color=COLOR_2)
        # Replaced MathTex with Text to avoid LaTeX dependency errors
        label2 = Text("(0, 5)", font_size=20, color=WHITE).next_to(dot2, UP, buff=0.1)
        
        self.play(
            FadeIn(dot2),
            FadeIn(label2),
            radar_cat.animate.move_to(pos_0_5 + 0.4 * UP)
        )
        
        # Show calculation 'dy/dx = -0/5 = 0'
        # Replaced MathTex with Text to avoid LaTeX dependency errors
        calc2 = Text("dy/dx = -0/5 = 0", font_size=24, color=COLOR_2)
        self.place_at_grid(calc2, "A5")
        self.play(Write(calc2))
        
        # Draw a horizontal tangent line at the top (0, 5)
        tangent2 = Line(
            pos_0_5 + 1.0 * LEFT,
            pos_0_5 + 1.0 * RIGHT,
            color=COLOR_2
        )
        self.play(Create(tangent2))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Tangent lines perfectly match our calculated values.
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(COLOR_3)
        )
        
        # Visual validation via highlights
        self.play(
            Indicate(tangent1, color=COLOR_3, scale_factor=1.2),
            Indicate(tangent2, color=COLOR_3, scale_factor=1.2)
        )
        self.wait(2)
