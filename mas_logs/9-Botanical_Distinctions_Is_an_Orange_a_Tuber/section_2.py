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
        # Setup the layout with title and lecture lines
        self.setup_layout(
            "Prerequisite Knowledge: Basic Plant Anatomy",
            [
                "Plants have parts above and below the ground.",
                "Flowers and fruits help the plant reproduce.",
                "Roots and stems store nutrients for survival."
            ]
        )

        # === Animation for Lecture Line 1 ===
        # Visualizing the ground division using white dashed line
        # Issue 30: above_ground_text and below_ground_text repositioned for better visual grouping
        ground_line = DashedLine(
            self.grid["C1"], 
            self.grid["C6"], 
            color=WHITE
        )
        
        above_ground_text = Text("Above Ground", font_size=20, color=WHITE)
        self.place_at_grid(above_ground_text, "B2", scale_factor=0.8)
        
        below_ground_text = Text("Below Ground", font_size=20, color=WHITE)
        self.place_at_grid(below_ground_text, "D2", scale_factor=0.8)
        
        # Plant diagram base stem shifted to column 3 to accommodate label changes
        plant_stem = Line(
            self.grid["B3"], 
            self.grid["D3"], 
            color="#228B22", 
            stroke_width=6
        )
        
        self.play(
            Create(ground_line),
            Write(above_ground_text),
            Write(below_ground_text),
            Create(plant_stem),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Highlighting reproductive organs (Fruits)
        # Issue 28: repositioned fruit to B3 and label to B4 (with 0.7 scale) to keep label on screen
        self.play(self.lecture[1].animate.set_color("#00FFFF"))
        
        fruit_obj = Circle(radius=0.25, color="#00FFFF", fill_opacity=1)
        self.place_at_grid(fruit_obj, "B3")
        
        repr_label = Text("Reproductive Organ", font_size=18, color="#00FFFF")
        self.place_at_grid(repr_label, "B4", scale_factor=0.7)
        
        self.play(
            GrowFromCenter(fruit_obj),
            Write(repr_label),
            run_time=1
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Highlighting storage organs (Roots/Tubers)
        # Issue 29: repositioned vessel to E3 and label to E4 (with 0.7 scale) to prevent clipping
        self.play(self.lecture[2].animate.set_color("#DEB887"))
        
        # Draw storage structures below ground, adjusted for the new column 3 position
        main_root = Line(self.grid["D3"], self.grid["E3"], color="#DEB887", stroke_width=5)
        branch_l = Line(self.grid["D3"], self.grid["E2"], color="#DEB887", stroke_width=3)
        branch_r = Line(self.grid["D3"], self.grid["E4"], color="#DEB887", stroke_width=3)
        
        storage_vessel = Ellipse(width=0.7, height=0.4, color="#DEB887", fill_opacity=1)
        self.place_at_grid(storage_vessel, "E3")
        
        storage_label = Text("Storage Organ", font_size=18, color="#DEB887")
        self.place_at_grid(storage_label, "E4", scale_factor=0.7)
        
        self.play(
            Create(main_root),
            Create(branch_l),
            Create(branch_r),
            FadeIn(storage_vessel),
            Write(storage_label),
            run_time=1.5
        )
        self.wait(2)
