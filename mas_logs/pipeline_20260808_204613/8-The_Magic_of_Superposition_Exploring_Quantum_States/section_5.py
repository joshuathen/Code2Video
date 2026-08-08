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

class Section5Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Application and Conclusion", [
            "Superposition enables massive quantum parallelism.",
            "Explore all maze paths simultaneously.",
            "Exponentially faster than classical computing."
        ])
        
        # Define visuals
        # 1. Circuit diagram + Maze asset
        gate = Rectangle(width=1, height=1, color=BLUE)
        maze_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/maze.svg", color=WHITE)
        maze_asset.next_to(gate, UP, buff=0.1)
        line_in = Line(LEFT, RIGHT).next_to(gate, LEFT, buff=0)
        line_out = Line(LEFT, RIGHT).next_to(gate, RIGHT, buff=0)
        circuit = VGroup(gate, line_in, line_out, maze_asset)

        # 2. Particle/Signal
        dot = Dot(color=YELLOW)

        # 3. Result State
        result_text = Text("0 or 1", color=GREEN)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(BLUE)
        self.place_in_area(circuit, "B4", "D6", scale_factor=0.6)
        self.play(Create(circuit))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        dot.move_to(line_in.get_start())
        self.play(FadeIn(dot))
        self.play(dot.animate.move_to(line_out.get_end()), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(GREEN)
        self.place_at_grid(result_text, "F5", scale_factor=0.8)
        self.play(FadeIn(result_text))
        self.wait(2)
