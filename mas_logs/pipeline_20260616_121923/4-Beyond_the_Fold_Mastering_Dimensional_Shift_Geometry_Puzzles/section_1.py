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

class Section1Scene(TeachingScene):
    def construct(self):
        # Initial Setup
        title = "The Ant's Perspective: Understanding 2D vs. 3D"
        lines = [
            'Meet Alpha, an ant living in a 2D world.',
            'Alpha sees objects only as flat lines.',
            "A sphere passes through Alpha's flat plane.",
            'A point grows into a line, then shrinks.',
            'Dimension shifts reveal parts of a hidden whole.'
        ]
        self.setup_layout(title, lines)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(ORANGE)
        
        # 2D World (Gray Rectangle)
        world_rect = Rectangle(width=5, height=4, fill_color="#808080", fill_opacity=0.2, stroke_color="#808080")
        self.place_in_area(world_rect, "A1", "F6")
        
        # Alpha (Orange Circle) - Fixed Issue 28: moved to C2 and scaled down
        alpha = Circle(radius=0.15, color="#FFA500", fill_opacity=1)
        self.place_at_grid(alpha, "C2", scale_factor=0.8)
        alpha_label = Text("Alpha", font_size=18, color="#FFA500")
        alpha_label.next_to(alpha, UP, buff=0.1)
        
        self.play(FadeIn(world_rect))
        self.play(Create(alpha), Write(alpha_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(RED)
        
        # Red Circle - Fixed Issue 29: moved to C4
        red_circle = Circle(radius=0.7, color="#FF0000")
        self.place_at_grid(red_circle, "C4")
        
        # Highlighting a segment - Fixed Issue 29: moved to C4
        highlight_arc = Arc(radius=0.7, start_angle=PI/4, angle=PI/2, color=YELLOW, stroke_width=6)
        self.place_at_grid(highlight_arc, "C4")
        
        self.play(Create(red_circle))
        self.play(Create(highlight_arc))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(GREEN)
        
        # Prepare for the cross-section animation
        self.play(FadeOut(red_circle), FadeOut(highlight_arc))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(GREEN)
        
        # Green circle expanding and shrinking
        green_circle = Circle(radius=0.01, color="#00FF00", fill_opacity=0.4)
        self.place_at_grid(green_circle, "D4")
        
        self.play(green_circle.animate.set_width(1.8), run_time=1.5)
        self.play(green_circle.animate.set_width(0.01), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(WHITE)
        
        # Fixed Issue 30: Using place_in_area for better label layout
        cross_section_label = Text("3D Sphere Cross-section", font_size=20, color="#FFFFFF")
        self.place_in_area(cross_section_label, "E3", "E5", scale_factor=0.7)
        
        # Re-expand green circle to show the result
        self.play(green_circle.animate.set_width(1.2), FadeIn(cross_section_label))
        self.wait(2)
