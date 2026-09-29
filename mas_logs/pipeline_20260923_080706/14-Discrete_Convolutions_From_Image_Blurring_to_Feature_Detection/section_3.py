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
            "Different kernels extract unique visual information.",
            "Identity kernels preserve the original image data.",
            "Edge detection kernels highlight sudden intensity jumps."
        ]
        self.setup_layout("Visualizing Operations: Filters as Feature Detectors", lecture_lines)
        
        # Assets
        camera_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/camera.svg", color=RED)
        photograph_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/photograph.svg", color=GREEN)
        
        # Kernel
        kernel = Square(side_length=1, color=RED).set_fill(RED, opacity=0.3)
        kernel_label = Text("Vertical Line Kernel", font_size=20, color=RED)
        kernel_group = VGroup(kernel, camera_icon, kernel_label).arrange(DOWN)
        
        # Image / Edge
        image = Rectangle(height=2, width=2, color=WHITE).set_fill(GRAY, opacity=0.5)
        edge_line = Line(UP, DOWN, color=YELLOW)
        edge_detection_visual = VGroup(image, edge_line)
        
        # Spike
        star_icon = photograph_icon.copy()
        
        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(self.lecture[0]))
        self.place_in_area(kernel_group, 'B4', 'C6', scale_factor=0.9)
        self.play(Create(kernel_group))
        self.lecture[0].set_color("#FF0000")

        # === Animation for Lecture Line 2 ===
        self.play(FadeIn(self.lecture[1]))
        self.place_at_grid(edge_detection_visual, 'D4', scale_factor=1.0)
        self.play(FadeIn(edge_detection_visual))
        self.lecture[1].set_color("#FFFFFF")

        # === Animation for Lecture Line 3 ===
        self.play(FadeIn(self.lecture[2]))
        # Slide effect
        self.play(kernel_group.animate.move_to(self.grid["D4"]), run_time=2)
        # Highlight spike
        self.place_at_grid(star_icon, 'D6', scale_factor=0.7)
        self.play(Flash(self.grid["D4"], color=GREEN), FadeIn(star_icon))
        self.lecture[2].set_color("#00FF00")
        
        self.wait(2)
