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

class Section4Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Real-World Application", [
            "Apply this to logistics and signal processing systems.",
            "Error margins aggregate in complex, multi-stage workflows.",
            "We calculate the total probability of timely delivery."
        ])

        # Define elements
        # Visualization: 2D kernel/pixel grid concept using Asset
        microchip = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/microchip.svg")
        camera = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/camera.svg")
        
        pixels = VGroup(*[Square(side_length=0.5, stroke_width=1, fill_opacity=0.3, color=BLUE) for _ in range(9)])
        pixels.arrange_in_grid(3, 3, buff=0)
        kernel = Square(side_length=0.5, stroke_width=2, color=YELLOW, fill_opacity=0)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        # Using asset
        self.place_at_grid(microchip, 'B4', scale_factor=0.5)
        self.add(microchip)
        # Grid adjusted for constraints
        self.place_in_area(pixels, 'C3', 'E5', scale_factor=0.6) 
        self.add(pixels)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(YELLOW)
        # Kernel adjusted
        self.place_at_grid(kernel, 'C3', scale_factor=0.9)
        self.add(kernel)
        
        # Animate filter scanning
        positions = ['C3', 'C4', 'C5', 'D3', 'D4', 'D5', 'E3', 'E4', 'E5']
        for pos in positions:
            self.play(kernel.animate.move_to(self.grid[pos]), run_time=0.3)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(YELLOW)
        # Using asset camera and adjusted result_rect
        result_rect = Rectangle(width=1.5, height=1.5, color="#00FF00", fill_opacity=0.5)
        self.place_at_grid(camera, 'B6', scale_factor=0.5)
        self.place_in_area(result_rect, 'C3', 'E5', scale_factor=0.8)
        self.play(FadeIn(camera), FadeIn(result_rect))
        self.wait(2)
