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

class Section7Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "Change of basis is a tool for changing perspective.",
            "It makes complex problems much easier to solve.",
            "This concept powers technologies like JPEG image compression."
        ]
        self.setup_layout("Summary and Real-world Application", lecture_lines)
        
        # Colors for matching
        COLOR_LINE_1 = BLUE
        COLOR_LINE_2 = GREEN
        COLOR_LINE_3 = ORANGE
        
        # === Animation for Lecture Line 1 ===
        # Standard vs Custom Perspective
        self.lecture[0].set_color(COLOR_LINE_1)
        
        # Create a mini standard grid
        std_grid = NumberPlane(
            x_range=[-2, 2, 1],
            y_range=[-2, 2, 1],
            background_line_style={"stroke_color": BLUE, "stroke_width": 1},
            axis_config={"stroke_color": BLUE_E}
        ).scale(0.4)
        std_label = Text("Standard", font_size=16, color=BLUE)
        std_group = VGroup(std_grid, std_label).arrange(DOWN, buff=0.2)
        # Position in top-left part of the grid area (A1-C3)
        self.place_in_area(std_group, "A1", "C3", scale_factor=1.0)
        
        # Create a mini custom grid (rotated/skewed)
        custom_grid = NumberPlane(
            x_range=[-2, 2, 1],
            y_range=[-2, 2, 1],
            background_line_style={"stroke_color": YELLOW, "stroke_width": 1},
            axis_config={"stroke_color": YELLOW_E}
        ).scale(0.4).apply_matrix([[1, 0.5], [0.3, 1]])
        custom_label = Text("Custom", font_size=16, color=YELLOW)
        custom_group = VGroup(custom_grid, custom_label).arrange(DOWN, buff=0.2)
        # Position in bottom-left part of the grid area (D1-F3)
        self.place_in_area(custom_group, "D1", "F3", scale_factor=1.0)
        
        self.play(Create(std_group), Create(custom_group))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Icons for Physics, Computer Graphics, Data Compression
        self.lecture[1].set_color(COLOR_LINE_2)
        
        # Physics Icon - Moved to B6 per Issue 52
        physics_box = RoundedRectangle(corner_radius=0.1, height=0.6, width=1.2, color=GREEN)
        physics_text = Text("Physics", font_size=14, color=GREEN)
        physics_icon = VGroup(physics_box, physics_text)
        self.place_at_grid(physics_icon, "B6", scale_factor=1.0)
        
        # Computer Graphics Icon with Asset (Issue 32) - Moved to D6 per Issue 51
        camera_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/camera.svg").scale(0.3).set_color(RED)
        graphics_text = Text("Graphics", font_size=14, color=RED)
        graphics_icon = VGroup(camera_asset, graphics_text).arrange(DOWN, buff=0.1)
        self.place_at_grid(graphics_icon, "D6", scale_factor=1.0)
        
        # Data Compression Icon - Moved to F6 per Issue 51
        compression_box = RoundedRectangle(corner_radius=0.1, height=0.6, width=1.2, color=ORANGE)
        compression_text = Text("Data", font_size=14, color=ORANGE)
        compression_icon = VGroup(compression_box, compression_text)
        self.place_at_grid(compression_icon, "F6", scale_factor=1.0)
        
        self.play(
            FadeIn(physics_icon),
            FadeIn(graphics_icon),
            FadeIn(compression_icon),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # JPEG and Final Summary
        self.lecture[2].set_color(COLOR_LINE_3)
        
        # JPEG logo - Moved to B5 per Issue 52
        jpeg_box = Square(side_length=1.0, color=WHITE, fill_opacity=0.1)
        jpeg_text = Text("JPEG", font_size=20, color=WHITE)
        jpeg_logo = VGroup(jpeg_box, jpeg_text)
        self.place_at_grid(jpeg_logo, "B5", scale_factor=1.0)
        
        # Final Summary - Adjusted area to D4-F5 per Issue 50
        final_summary = Text("Change of Basis:\nA new way to see space.", font_size=20, color=PURPLE)
        self.place_in_area(final_summary, "D4", "F5", scale_factor=1.0)
        
        self.play(Write(jpeg_logo))
        self.play(FadeIn(final_summary, shift=UP))
        self.wait(3)
