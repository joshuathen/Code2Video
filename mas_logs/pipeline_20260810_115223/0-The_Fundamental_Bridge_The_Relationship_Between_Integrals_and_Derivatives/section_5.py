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
        self.setup_layout("Synthesis & Summary", [
            "Differentiation and integration are linked inverses.",
            "They represent the cycle of rates and accumulation.",
            "Mastery allows navigation between these two worlds."
        ])
        
        # Create visual elements: The Cycle of Calculus
        diff_node = VGroup(Circle(radius=0.5, color=BLUE), Text("d/dx", font_size=20))
        int_node = VGroup(Circle(radius=0.5, color=GREEN), Text("∫", font_size=30))
        
        arrow1 = CurvedArrow(start_point=LEFT*0.5, end_point=RIGHT*0.5, angle=-PI/2, color=YELLOW)
        arrow2 = CurvedArrow(start_point=RIGHT*0.5, end_point=LEFT*0.5, angle=-PI/2, color=YELLOW)
        
        # Grouped cycle
        cycle = VGroup(diff_node, int_node, arrow1, arrow2)
        diff_node.next_to(arrow1, LEFT)
        int_node.next_to(arrow1, RIGHT)
        arrow2.next_to(arrow1, DOWN)
        
        # Position using corrected instructions for issue 32
        self.place_in_area(cycle, 'B4', 'E6', scale_factor=0.9)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(BLUE)
        self.play(FadeIn(cycle))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(GREEN)
        self.play(Indicate(cycle))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(YELLOW)
        self.play(FadeOut(cycle))
        self.wait(1)
