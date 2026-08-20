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
        self.setup_layout("Real-World Application: Image Processing & AI", 
                          ["CNNs stack these convolution operations.", 
                           "They recognize complex, layered objects.", 
                           "Simple features become complex images."])
        
        # Load assets
        lens_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/lens.svg")
        computer_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/computer.svg")

        # 1. Convolutional Neural Network layer diagram concept
        # Representing a small 3x3 kernel and output
        grid_square = Square(side_length=0.6, color=WHITE).set_fill(GREY, opacity=0.3)
        kernel_base = VGroup(*[grid_square.copy() for _ in range(9)]).arrange_in_grid(3, 3, buff=0.05)
        kernel = VGroup(kernel_base, lens_icon.scale(0.5)).arrange(DOWN, buff=0.1)
        self.place_at_grid(kernel, 'B4', scale_factor=0.8)

        # Output activation map
        output_map_base = VGroup(*[Square(side_length=0.4, color=WHITE).set_fill(BLACK, opacity=0.8) for _ in range(16)]).arrange_in_grid(4, 4, buff=0.05)
        output_map = VGroup(output_map_base, computer_icon.scale(0.5)).arrange(DOWN, buff=0.1)
        self.place_at_grid(output_map, 'F4', scale_factor=0.8)
        
        self.add(kernel, output_map)

        # === Animation for Lecture Line 1 ===
        # CNNs stack these convolution operations.
        # Highlight kernel filter weights
        self.play(kernel.animate.set_color("#00CED1"), run_time=1.0)
        self.play(self.lecture[0].animate.set_color("#00CED1"), run_time=0.5)

        # === Animation for Lecture Line 2 ===
        # They recognize complex, layered objects.
        # Highlight activation map
        self.play(output_map.animate.set_color("#FF8C00"), run_time=1.0)
        self.play(self.lecture[1].animate.set_color("#FF8C00"), run_time=0.5)
        
        # === Animation for Lecture Line 3 ===
        # Simple features become complex images.
        self.play(self.lecture[2].animate.set_color(YELLOW), run_time=0.5)
        self.wait(1)
