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

class Section4Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Synthesis & Visual Summary", ["Limits define function trends.", "Precision requires formal proof.", "L'Hôpital simplifies complex calculations."])
        
        # Create summary board elements
        # Propagating colors as per instructions (B039)
        concept_1 = Text("1. Intuition (Trends)", font_size=24, color="#3498db") # Blue
        concept_2 = Text("2. Epsilon-Delta (Proof)", font_size=24, color="#f1c40f") # Yellow
        concept_3 = Text("3. L'Hôpital (Calculation)", font_size=24, color="#e74c3c") # Red
        
        board = VGroup(concept_1, concept_2, concept_3).arrange(DOWN, buff=0.5)
        # Applying fix: Move grid visual group per Critic #30
        self.place_in_area(board, "C4", "F6", scale_factor=0.7)
        
        # [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg] - Not a valid file path, dummy placeholder as per storyboard
        
        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(concept_1))
        self.lecture[0].set_color("#3498db")

        # === Animation for Lecture Line 2 ===
        self.play(FadeIn(concept_2))
        self.lecture[1].set_color("#f1c40f")

        # === Animation for Lecture Line 3 ===
        self.play(FadeIn(concept_3))
        self.lecture[2].set_color("#e74c3c")
        
        # Final flow connection visualization (Positioning in D, B040)
        arrow = Arrow(concept_1.get_bottom(), concept_3.get_top(), color=WHITE)
        self.add(arrow)
        
        self.wait(2)
        # [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg]
        self.play(FadeOut(self.lecture), FadeOut(board), FadeOut(arrow), FadeOut(self.title))
        self.wait(1)
