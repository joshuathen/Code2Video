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

class Section3Scene(TeachingScene):
    def construct(self):
        # Setup the layout with title and lecture lines
        self.setup_layout("What is an Orange? (Botanical Fruit)", [
            "An orange is botanically classified as a fruit.",
            "It develops from the flower's ovary after pollination.",
            "Inside, seeds wait to grow into new trees.",
            "Oranges grow high up in the air on branches.",
            "Their main job is spreading seeds for reproduction."
        ])

        # === Animation for Lecture Line 1 ===
        # A whole orange (#FFA500) appears at the top of the screen (right-side grid).
        orange_whole = Circle(radius=1.0, fill_opacity=1, color="#FFA500", stroke_width=0)
        self.place_in_area(orange_whole, "A3", "B4", scale_factor=0.6)
        self.play(self.lecture[0].animate.set_color("#FFA500"), FadeIn(orange_whole))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # The orange icon splits to reveal a cross-section with juice segments (#FFD700).
        cross_section = VGroup()
        outer_rim = Circle(radius=1.0, fill_opacity=1, color="#FFA500", stroke_width=0)
        # Adding a white layer to represent the pith/rind transition
        pith = Circle(radius=0.96, fill_opacity=1, color="#FFFFFF", stroke_width=0)
        segments = VGroup(*[
            AnnularSector(inner_radius=0.1, outer_radius=0.9, angle=TAU/10 - 0.1, 
                         start_angle=i*TAU/10 + 0.05, color="#FFD700")
            for i in range(10)
        ])
        cross_section.add(outer_rim, pith, segments)
        # Position in middle-center of the right grid panel
        # Fix Issue 33: Scale reduced from 1.1 to 0.9
        self.place_in_area(cross_section, "B3", "D4", scale_factor=0.9)
        
        self.play(
            self.lecture[1].animate.set_color("#FFD700"), 
            ReplacementTransform(orange_whole, cross_section)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Small white seeds (#FFFFFF) flash and are highlighted within the segments.
        seeds = VGroup()
        center = cross_section.get_center()
        # Place 5 seeds in alternating segments
        for i in range(5):
            angle = (i * 2 + 1) * TAU / 10
            seed = Dot(color="#FFFFFF", radius=0.06)
            # Offset position relative to center
            seed.move_to(center + 0.4 * np.array([np.cos(angle), np.sin(angle), 0]))
            seeds.add(seed)

        self.play(self.lecture[2].animate.set_color("#FFFFFF"), FadeIn(seeds))
        # Highlighting seeds with individual flashes
        self.play(*[Flash(seed, color="#FFFFFF", line_length=0.2) for seed in seeds])
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # An arrow (#FFFFFF) points from the orange to a text label 'Aerial Growth' (#00FFFF).
        aerial_label = Text("Aerial Growth", font_size=24, color="#00FFFF")
        # Fix Issue 32: Moved from F5 to area D5-E6 and scaled to 0.8
        self.place_in_area(aerial_label, 'D5', 'E6', scale_factor=0.8)
        
        # Calculate arrow from orange side to label top
        arrow = Arrow(
            start=cross_section.get_edge_center(DOWN + RIGHT), 
            end=aerial_label.get_top(), 
            color="#FFFFFF", 
            buff=0.1
        )
        
        self.play(
            self.lecture[3].animate.set_color("#00FFFF"), 
            Create(arrow), 
            Write(aerial_label)
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # The label 'Botanical Fruit' (#00FF00) appears in bold.
        fruit_label = Text("Botanical Fruit", font_size=36, color="#00FF00", weight=BOLD)
        # Fix Issue 31: Moved from E3 to area E1-F2 and scaled to 0.8
        self.place_in_area(fruit_label, 'E1', 'F2', scale_factor=0.8)
        
        self.play(self.lecture[4].animate.set_color("#00FF00"), Write(fruit_label))
        self.wait(2)
