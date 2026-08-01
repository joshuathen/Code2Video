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
        # Initial layout setup
        lecture_lines = [
            "A tuber is a swollen, underground part of a stem.",
            "It acts as a massive energy storage tank.",
            "Potatoes use these \"eyes\" to sprout new clones.",
            "They stay hidden in the soil to survive winter.",
            "Tubers focus on storage, not on spreading seeds."
        ]
        self.setup_layout("What is a Tuber? (Storage Organs)", lecture_lines)

        # === Animation for Lecture Line 1 ===
        # A tuber is a swollen, underground part of a stem.
        self.lecture[0].set_color("#D2B48C")
        
        # Soil reference line
        soil_line = Line(self.grid['B1'], self.grid['B6'], color=GREY_E)
        
        # Underground stem connecting from above soil to the tuber
        stem = Rectangle(width=0.2, height=2.0, color="#8B4513", fill_opacity=1, stroke_width=0)
        self.place_in_area(stem, 'A3', 'C3')
        
        # Swollen potato body
        potato = Ellipse(width=3.5, height=2.5, color="#D2B48C", fill_opacity=1, stroke_color=WHITE, stroke_width=1)
        self.place_in_area(potato, 'C2', 'E5')
        
        self.play(Create(soil_line), Create(stem))
        self.play(FadeIn(potato))

        # === Animation for Lecture Line 2 ===
        # It acts as a massive energy storage tank.
        self.lecture[1].set_color("#FFFFFF")
        
        # Energy storage symbols (starch hexagons)
        hex1 = RegularPolygon(6, color="#FFFFFF", fill_opacity=0.8).scale(0.2)
        hex2 = RegularPolygon(6, color="#FFFFFF", fill_opacity=0.8).scale(0.2)
        self.place_at_grid(hex1, 'D3')
        self.place_at_grid(hex2, 'D4')
        
        # Pulse the potato to "reveal" contents
        self.play(potato.animate.scale(1.1), rate_func=there_and_back)
        self.play(FadeIn(hex1), FadeIn(hex2))

        # === Animation for Lecture Line 3 ===
        # Potatoes use these "eyes" to sprout new clones.
        self.lecture[2].set_color("#5C4033")
        
        # Buds/Eyes on the surface
        bud1 = Dot(color="#5C4033").scale(1.5)
        bud2 = Dot(color="#5C4033").scale(1.5)
        bud3 = Dot(color="#5C4033").scale(1.5)
        self.place_at_grid(bud1, 'C4')
        self.place_at_grid(bud2, 'E3')
        self.place_at_grid(bud3, 'D5', scale_factor=0.7)
        
        bud_label = Text("Buds", font_size=24, color="#5C4033")
        self.place_at_grid(bud_label, 'C6', scale_factor=0.8)
        
        self.play(FadeIn(bud1), FadeIn(bud2), FadeIn(bud3))
        self.play(Write(bud_label))

        # === Animation for Lecture Line 4 ===
        # They stay hidden in the soil to survive winter.
        self.lecture[3].set_color("#8B4513")
        
        # Emphasize being underground with a soil overlay
        soil_overlay = Rectangle(width=5.0, height=4.0, color="#8B4513", fill_opacity=0.15, stroke_width=0)
        self.place_in_area(soil_overlay, 'B1', 'F6')
        self.play(FadeIn(soil_overlay))

        # === Animation for Lecture Line 5 ===
        # Tubers focus on storage, not on spreading seeds.
        self.lecture[4].set_color("#00FF00")
        
        # Final identifying label
        storage_label = Text("Storage Tuber", font_size=28, color="#00FF00", weight=BOLD)
        self.place_in_area(storage_label, 'F3', 'F4', scale_factor=0.8)
        
        self.play(Write(storage_label))
        self.wait(2)
