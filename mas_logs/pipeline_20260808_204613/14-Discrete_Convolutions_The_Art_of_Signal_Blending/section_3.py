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

class Section3Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "Filters act as local processing windows.",
            "Slide the kernel over image pixels.",
            "Kernel values modify neighboring pixel data."
        ]
        self.setup_layout("Visualizing Convolution: The 'Blur' Filter", lecture_lines)
        
        # Assets
        photo_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/photograph.svg")
        camera_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/camera.svg")
        
        # Grid visual
        img_grid = VGroup(*[Square(side_length=0.7).set_stroke(WHITE, 1) for _ in range(9)])
        img_grid.arrange_in_grid(3, 3, buff=0)
        self.place_in_area(img_grid, 'B3', 'E5', scale_factor=0.7)
        
        # Kernel
        kernel = Square(side_length=0.7).set_stroke(RED, 2).set_fill(RED, opacity=0.3)
        self.place_at_grid(kernel, 'C3', scale_factor=0.7)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFFF00"))
        self.place_at_grid(photo_asset, 'A6', scale_factor=0.3)
        self.play(Create(img_grid), FadeIn(photo_asset))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FFFF00"))
        self.play(kernel.animate.move_to(self.grid['C4']))
        self.wait(0.5)
        self.play(kernel.animate.move_to(self.grid['D4']))
        self.wait(0.5)
        self.play(kernel.animate.move_to(self.grid['D3']))
        self.wait(0.5)
        self.play(kernel.animate.move_to(self.grid['C3']))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FFFF00"))
        self.place_at_grid(camera_asset, 'F6', scale_factor=0.3)
        result_pixels = VGroup(*[Dot(color=BLUE).move_to(self.grid[pos]) for pos in ['B3', 'B4', 'B5', 'C3', 'C4', 'C5', 'D3', 'D4', 'D5']])
        self.play(FadeIn(result_pixels), FadeIn(camera_asset))
        self.wait(2)
