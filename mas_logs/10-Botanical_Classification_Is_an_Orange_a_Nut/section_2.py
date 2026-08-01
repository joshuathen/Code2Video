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

class Section2Scene(TeachingScene):
    def construct(self):
        # Setup layout with title and lecture lines
        title_text = "Prerequisite Knowledge: What is a Fruit?"
        lecture_lines = [
            "Botanically, fruits develop from a plant's flower.",
            "The ovary wall transforms into the protective pericarp.",
            "Fruits are categorized as either fleshy or dry."
        ]
        self.setup_layout(title_text, lecture_lines)

        # === Animation for Lecture Line 1 ===
        # Color line 1 to pink to match the starting flower petals
        self.play(self.lecture[0].animate.set_color("#FF69B4"))
        
        # Create flower components: Petals (#FF69B4) and Ovary (#00FF00)
        # Arrange 6 petals around the central ovary
        petals = VGroup(*[Circle(radius=0.4, color="#FF69B4", fill_opacity=0.7) for _ in range(6)])
        for i, petal in enumerate(petals):
            angle = i * 60 * DEGREES
            petal.shift(0.5 * np.array([np.cos(angle), np.sin(angle), 0]))
            
        ovary = Circle(radius=0.3, color="#00FF00", fill_opacity=1)
        flower = VGroup(petals, ovary)
        
        # Place the flower in the center of the upper animation grid (Issue 37)
        self.place_in_area(flower, "A3", "C4", scale_factor=0.7)
        
        self.play(FadeIn(flower))
        self.wait(0.5)
        
        # Highlight the green Ovary with a pulse effect
        self.play(ovary.animate.scale(1.2), run_time=0.4)
        self.play(ovary.animate.scale(1/1.2), run_time=0.4)
        self.wait(0.5)
        
        # Transformation: The pink flower petals wither, and the Ovary transforms into the fruit (#FF4500)
        fruit_shape = Circle(radius=0.8, color="#FF4500", fill_opacity=0.9, stroke_width=4)
        # Maintain hierarchical layout (Issue 37)
        self.place_in_area(fruit_shape, "A3", "C4", scale_factor=0.7)
        
        self.play(
            FadeOut(petals),
            Transform(ovary, fruit_shape),
            run_time=1.5
        )
        # The 'ovary' mobject now represents the fruit visual
        fruit_obj = ovary 
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Color line 2 to match the pericarp label (#ADFF2F)
        self.play(self.lecture[1].animate.set_color("#ADFF2F"))
        
        # Highlight the Pericarp (the wall of the fruit)
        pericarp_highlight = fruit_obj.copy().set_fill(opacity=0).set_stroke(color="#ADFF2F", width=10)
        pericarp_label = Text("Pericarp", font_size=24, color="#ADFF2F")
        # Align label with the new object position (Issue 39)
        self.place_at_grid(pericarp_label, "B5", scale_factor=0.9)
        
        self.play(Create(pericarp_highlight))
        self.play(Write(pericarp_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Color line 3 to match the 'Fleshy' branch color (#FF8C00)
        self.play(self.lecture[2].animate.set_color("#FF8C00"))
        
        # Move the fruit group to the top for the flowchart view
        fruit_group = VGroup(fruit_obj, pericarp_highlight, pericarp_label)
        self.play(fruit_group.animate.scale(0.6).move_to(self.grid["B3"]), run_time=1)
        
        # Define flowchart categories: Fleshy (#FF8C00) and Dry (#DEB887)
        fleshy_label = Text("Fleshy", font_size=26, color="#FF8C00")
        dry_label = Text("Dry", font_size=26, color="#DEB887")
        
        # Shift labels to the E row for vertical separation (Issue 38)
        self.place_at_grid(fleshy_label, "E2")
        self.place_at_grid(dry_label, "E4")
        
        # Draw arrows branching from the fruit (C3 area) to the labels
        arrow_fleshy = Arrow(start=self.grid["C3"], end=self.grid["E2"], color=WHITE, buff=0.2)
        arrow_dry = Arrow(start=self.grid["C3"], end=self.grid["E4"], color=WHITE, buff=0.2)
        
        self.play(
            Create(arrow_fleshy),
            Create(arrow_dry),
            Write(fleshy_label),
            Write(dry_label)
        )
        self.wait(2)
