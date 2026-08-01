from manim import *

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

class Section6Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Modern Applications: From Maps to Data", [
            "These curves flatten 2D images into 1D data.",
            "Preserving neighbor pixels significantly improves image compression.",
            "Databases use this for efficient range queries.",
            "Google Maps indexes locations using space-filling curves.",
            "Abstract math solves real-world high-dimensional data problems."
        ])

        # Define indices for a 4x4 Hilbert Curve on the grid (Order 2)
        # Mapping (x, y) coordinates to grid cell IDs
        # x: 0->2, 1->3, 2->4, 3->5 | y: 0->E, 1->D, 2->C, 3->B
        path_coords = [
            "E2", "D2", "D3", "E3", 
            "E4", "E5", "D5", "D4", 
            "C4", "C5", "B5", "B4", 
            "B3", "B2", "C2", "C3"
        ]

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#ADD8E6"))
        
        # Draw 4x4 Grid of "Pixels"
        pixels = VGroup()
        for coord in path_coords:
            p = Square(side_length=0.7, fill_opacity=0.3, color=BLUE, stroke_width=2)
            self.place_at_grid(p, coord)
            pixels.add(p)
        
        self.play(Create(pixels))

        # Draw Hilbert Curve Path
        hilbert_path = VMobject(color=WHITE)
        points = [self.grid[coord] for coord in path_coords]
        hilbert_path.set_points_as_corners(points)
        
        self.play(Create(hilbert_path), run_time=2)

        # 1D Flattening Row (using area F2-F5 for better alignment with the 4x4 grid above)
        one_d_row = VGroup(*[Square(side_length=0.25, fill_opacity=0.5, color=BLUE) for _ in range(16)])
        one_d_row.arrange(RIGHT, buff=0.1)
        self.place_in_area(one_d_row, "F2", "F5")
        
        # Show data flattening (ghost transform)
        self.play(ReplacementTransform(pixels.copy(), one_d_row), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#90EE90"))
        
        # Highlight a 2x2 block (indices 0-3: E2, D2, D3, E3)
        block_2x2 = VGroup(pixels[0], pixels[1], pixels[2], pixels[3])
        block_highlight = SurroundingRectangle(block_2x2, color="#90EE90", buff=0.1)
        
        one_d_subset = VGroup(*one_d_row[0:4])
        one_d_highlight = SurroundingRectangle(one_d_subset, color="#90EE90", buff=0.05)
        
        self.play(Create(block_highlight), Create(one_d_highlight))
        self.play(
            block_2x2.animate.set_fill("#90EE90", opacity=0.8), 
            one_d_subset.animate.set_fill("#90EE90", opacity=0.8)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#F08080"))
        
        # Highlight a range in 1D and show its 2D area (indices 6-9 -> D5, D4, C4, C5)
        range_subset_1d = VGroup(*one_d_row[6:10])
        range_subset_2d = VGroup(pixels[6], pixels[7], pixels[8], pixels[9])
        
        range_highlight_1d = SurroundingRectangle(range_subset_1d, color="#F08080", buff=0.05)
        range_highlight_2d = SurroundingRectangle(range_subset_2d, color="#F08080", buff=0.1)

        self.play(
            FadeOut(block_highlight), FadeOut(one_d_highlight),
            Create(range_highlight_1d), Create(range_highlight_2d)
        )
        self.play(
            range_subset_1d.animate.set_fill("#F08080", opacity=0.8),
            range_subset_2d.animate.set_fill("#F08080", opacity=0.8)
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color("#FFFFE0"))
        
        # Map dots appearing on the grid (Google Maps concept)
        map_dots = VGroup()
        dot_locations = ["B2", "C5", "E3", "D4"]
        for loc in dot_locations:
            dot = Dot(color="#FFFFE0").scale(1.5)
            self.place_at_grid(dot, loc)
            map_dots.add(dot)
            
        self.play(FadeIn(map_dots, shift=UP))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color("#FFB6C1"))
        
        summary_text = Text("High-Dimensional Efficiency", font_size=24, color="#FFB6C1")
        self.place_in_area(summary_text, "A2", "A5", scale_factor=0.8)
        
        self.play(Write(summary_text))
        self.play(Indicate(summary_text))
        self.wait(2)
