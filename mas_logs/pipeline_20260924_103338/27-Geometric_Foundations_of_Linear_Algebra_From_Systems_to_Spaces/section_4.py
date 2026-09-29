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
        lecture_lines = [
            "Null space contains vectors mapped to the origin.",
            "These are directions squashed into zero space.",
            "Think of the security camera's hidden line."
        ]
        self.setup_layout("Null Space: The 'Hidden' Kernel", lecture_lines)
        
        # Define objects
        plane = Axes(x_range=[-3, 3], y_range=[-3, 3], axis_config={"include_tip": True}).scale(0.5)
        vector_v = Vector([1, 1], color="#FF5733")
        origin_dot = Dot(color=WHITE)
        kernel_label = Text("Kernel", color="#33FF57", font_size=20)
        null_space_line = Line(start=[-2, -2, 0], end=[2, 2, 0], color="#FFFF33", stroke_width=4)
        
        # Asset integration
        camera_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/camera.svg")

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FF5733")
        self.place_at_grid(plane, "B3", scale_factor=0.8)
        self.place_at_grid(origin_dot, "B3", scale_factor=1.2)
        self.play(Create(plane), GrowFromCenter(origin_dot))
        self.play(GrowArrow(vector_v.move_to(self.grid["B3"] + np.array([-0.5, -0.5, 0]))))

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#33FF57")
        self.place_at_grid(kernel_label, "C2", scale_factor=0.7)
        self.play(FadeIn(kernel_label))
        self.play(vector_v.animate.move_to(self.grid["B3"]))

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FFFF33")
        self.place_at_grid(null_space_line, "B3", scale_factor=0.8)
        self.place_at_grid(camera_icon, "A3", scale_factor=0.5)
        self.play(Create(null_space_line), FadeIn(camera_icon))
        self.wait(2)
