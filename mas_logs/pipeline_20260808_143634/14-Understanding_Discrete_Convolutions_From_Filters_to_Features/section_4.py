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
        lecture_lines = ["Convolution helps AI identify complex shapes.", "Edges are detected by specific filter patterns.", "Abstract features emerge from these local details."]
        self.setup_layout("Real-World Application: Feature Extraction", lecture_lines)
        
        # Create image grid
        grid_square = Square(side_length=0.7, color="#FFFFFF")
        image_grid = VGroup(*[grid_square.copy() for _ in range(9)]).arrange_in_grid(3, 3, buff=0)
        
        # Asset integration
        camera_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/camera.svg", color=WHITE)
        camera_icon.scale(0.5).next_to(image_grid, UP)
        
        # Create kernel
        kernel = Square(side_length=0.7, color="#FF5733")
        filter_label = Text("Filter", font_size=16, color="#FF5733")
        
        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(image_grid), FadeIn(camera_icon))
        self.place_at_grid(image_grid, "B3", scale_factor=0.75)
        self.lecture[0].set_color("#FF5733")

        # === Animation for Lecture Line 2 ===
        self.play(FadeIn(kernel))
        self.place_at_grid(kernel, "E3", scale_factor=0.7)
        self.place_at_grid(filter_label, "D3", scale_factor=0.6)
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color("#FF5733")

        # === Animation for Lecture Line 3 ===
        activation_highlight = Square(side_length=0.7, color="#FFFF00", fill_opacity=0.5).move_to(image_grid[4])
        microscope_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/microscope.svg", color="#FFFF00")
        microscope_icon.scale(0.5).next_to(activation_highlight, RIGHT)
        self.play(FadeIn(activation_highlight), FadeIn(microscope_icon))
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color("#FFFF00")
        self.wait(2)
