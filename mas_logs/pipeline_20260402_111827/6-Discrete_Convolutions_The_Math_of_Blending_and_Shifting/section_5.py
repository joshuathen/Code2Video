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
        # Setup title and lecture lines
        title = "Application: Image Processing and Blurring"
        lines = [
            "In 2D, images are grids of discrete pixel values.",
            "A small matrix, or kernel, slides over these pixels.",
            "Blurring replaces pixels with a weighted average of neighbors.",
            "The Pixel Fox becomes soft as the kernel spreads data.",
            "Convolutional networks use this to identify complex features."
        ]
        self.setup_layout(title, lines)

        # Assets
        fox_asset_path = "/mmfs1/data/home/jthen/Code2Video/assets/icon/fox.svg"

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        
        # Show Fox Asset
        fox = SVGMobject(fox_asset_path, height=2, color=WHITE)
        self.place_at_grid(fox, "B3", scale_factor=0.8)
        self.play(FadeIn(fox))
        self.wait(1)

        # Create 5x5 grid
        pixel_grid = VGroup()
        pixel_values = VGroup()
        grid_data = np.random.randint(50, 255, size=(5, 5))
        
        # We'll map the 5x5 grid into rows B-F and cols 2-6
        grid_positions = [
            ["B2", "B3", "B4", "B5", "B6"],
            ["C2", "C3", "C4", "C5", "C6"],
            ["D2", "D3", "D4", "D5", "D6"],
            ["E2", "E3", "E4", "E5", "E6"],
            ["F2", "F3", "F4", "F5", "F6"]
        ]

        for r in range(5):
            for c in range(5):
                sq = Square(side_length=0.9, fill_opacity=0.3, color=GRAY)
                val = grid_data[r, c]
                sq.set_fill(color=interpolate_color(BLACK, WHITE, val/255.0))
                self.place_at_grid(sq, grid_positions[r][c])
                
                label = Text(str(val), font_size=16, color=WHITE)
                label.move_to(sq.get_center())
                
                pixel_grid.add(sq)
                pixel_values.add(label)

        self.play(
            FadeOut(fox),
            FadeIn(pixel_grid),
            FadeIn(pixel_values)
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)

        # 3x3 red kernel frame
        # Position it to cover top-left 3x3 (B2 to D4)
        kernel_rect = Rectangle(width=3.1, height=2.8, color=RED, stroke_width=4)
        self.place_in_area(kernel_rect, "B2", "D4")
        
        self.play(Create(kernel_rect))
        self.wait(0.5)
        
        # Slide it to the center 3x3 (C3 to E5)
        center_pos = self.grid["D4"] # roughly center of C3-E5
        self.play(kernel_rect.animate.move_to(center_pos))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)

        # Highlight neighbors in the 3x3 frame
        center_sq = pixel_grid[12] # center of 25
        center_val_label = pixel_values[12]
        
        # Calculate average (symbolic)
        avg_val = 150 # example average
        new_color = interpolate_color(BLACK, WHITE, avg_val/255.0)

        self.play(
            Indicate(kernel_rect),
            center_sq.animate.set_fill(new_color, opacity=0.8),
            run_time=1.5
        )
        
        new_label = Text(str(avg_val), font_size=16, color=YELLOW).move_to(center_sq.get_center())
        self.play(Transform(center_val_label, new_label))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)

        # Change kernel to edge detection and make edges glow white
        edge_kernel_text = Text("Edge Detection", font_size=20, color=RED)
        self.place_at_grid(edge_kernel_text, "A4")
        
        self.play(Write(edge_kernel_text))
        
        # Make edges (perimeter of the 5x5 grid) glow white
        glow_anims = []
        edge_indices = [0,1,2,3,4, 5,9, 10,14, 15,19, 20,21,22,23,24]
        for idx in edge_indices:
            glow_anims.append(pixel_grid[idx].animate.set_stroke(WHITE, width=6).set_fill(WHITE, opacity=0.9))
        
        self.play(*glow_anims)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)

        # Transform grid into Feature Map
        # In Manim Community Edition, TEAL is the standard cyan-like color.
        feature_map = SVGMobject(fox_asset_path, height=2.5, color=TEAL)
        self.place_in_area(feature_map, "B2", "F6")
        
        v_shape = VGroup(
            Line(self.grid["C3"], self.grid["E4"], color=TEAL, stroke_width=8),
            Line(self.grid["E4"], self.grid["C5"], color=TEAL, stroke_width=8)
        )

        self.play(
            FadeOut(pixel_grid),
            FadeOut(pixel_values),
            FadeOut(kernel_rect),
            FadeOut(edge_kernel_text),
            FadeIn(feature_map),
            Create(v_shape)
        )
        self.wait(2)
