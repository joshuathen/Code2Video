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
        lecture_lines = [
            "Information is measured by narrowing down possibilities.",
            "Surprise equates to the amount of information gained.",
            "Entropy quantifies the average surprise of a result.",
            "Finding rare outcomes yields high information.",
            "Common outcomes provide little new information."
        ]
        self.setup_layout("The Intuition: What is Information?", lecture_lines)
        
        # Assets (SVGs)
        box_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/box.svg"
        
        boxes = VGroup(*[SVGMobject(box_path).set_color("#FFD700") for _ in range(5)])
        q_marks = VGroup(*[Text("?", color="#FFD700", font_size=30) for _ in range(5)])
        
        # Assemble boxes
        box_elements = VGroup()
        for i in range(5):
            pair = VGroup(boxes[i], q_marks[i])
            box_elements.add(pair)
            self.place_at_grid(pair, f"B{i+1}", scale_factor=0.6)

        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(box_elements))
        self.lecture[0].set_color("#3498DB")
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Using VideoCritic fix for positioning
        self.place_in_area(box_elements, 'B2', 'B4', scale_factor=0.9)
        self.lecture[1].set_color("#F1C40F")
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Use placeholders for symbols as per storyboard
        symbol_rare = Text("★", color="#00FF00", font_size=40)
        symbol_common = Text("●", color="#FFFFFF", font_size=40)
        
        # VideoCritic fixes for C2, C3
        self.place_at_grid(symbol_rare, 'C2', scale_factor=0.7)
        self.place_at_grid(symbol_common, 'C3', scale_factor=0.7)
        
        self.play(FadeIn(symbol_rare), FadeIn(symbol_common))
        self.lecture[2].set_color("#2ECC71")
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        high_info_circle = Circle(color="#FF4500").surround(symbol_rare)
        self.play(Create(high_info_circle))
        self.lecture[3].set_color("#E74C3C")
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        final_text = Text("Information = Surprise", color=WHITE, font_size=30)
        self.place_in_area(final_text, 'A2', 'C5', scale_factor=0.8) # VideoCritic fix for Grid_Visual_Text
        self.play(FadeOut(VGroup(box_elements, symbol_rare, symbol_common, high_info_circle)), FadeIn(final_text))
        self.lecture[4].set_color("#E67E22")
        self.wait(2)
