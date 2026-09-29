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
            "Minimum moves equal two to the n minus one.",
            "Complexity grows exponentially with more disks.",
            "This puzzle demonstrates recursive algorithmic logic."
        ]
        self.setup_layout("Application and Complexity", lecture_lines)
        
        # Colors for lines
        colors = ["#00FFFF", "#FF00FF", "#FFFF00"]

        # === Animation for Lecture Line 1 ===
        # Visualize the growth 2^n - 1
        formula = MathTex(r"2^n - 1", font_size=40).set_color(colors[0])
        self.place_at_grid(formula, "B2", scale_factor=1.2)
        
        # Add asset disk icon
        disk_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/disk.svg")
        disk_icon.next_to(formula, RIGHT, buff=0.3)
        
        self.play(Write(formula), FadeIn(disk_icon))
        self.play(self.lecture[0].animate.set_color(colors[0]))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Show growth curve
        axes = Axes(x_range=[0, 6, 1], y_range=[0, 32, 8], axis_config={"include_tip": False})
        curve = axes.plot(lambda x: 2**x - 1, color=colors[1])
        graph = VGroup(axes, curve)
        self.place_in_area(graph, "E2", "F6", scale_factor=0.5)
        self.play(Create(graph))
        self.play(self.lecture[1].animate.set_color(colors[1]))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Highlight logic
        tag = Text("RECURSION", font_size=30, color=colors[2])
        self.place_at_grid(tag, "C6", scale_factor=0.8)
        self.play(FadeIn(tag, shift=UP))
        self.play(self.lecture[2].animate.set_color(colors[2]))
        self.wait(2)
