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
        self.setup_layout("Learning Rate: The Size of the Step", [
            "Learning rate dictates step size.",
            "Large steps might overshoot the valley.",
            "Tiny steps are inefficiently slow."
        ])
        
        # Define the graph
        axes = Axes(x_range=[-3, 3], y_range=[-1, 5], axis_config={"include_tip": False})
        curve = axes.plot(lambda x: x**2, color=BLUE)
        
        # Applying requested position fixes
        self.place_in_area(axes, 'D2', 'F6', scale_factor=0.3)
        self.place_in_area(curve, 'D2', 'F6', scale_factor=0.3)
        
        dot = Dot(color=YELLOW)
        dot.move_to(axes.c2p(-2.5, 6.25))
        self.add(dot)

        # Assets
        mountain = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/mountain.svg")
        valley = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/valley.svg")
        
        self.place_at_grid(mountain, 'B2', scale_factor=0.5)
        self.place_at_grid(valley, 'E5', scale_factor=0.5)
        
        # === Animation for Lecture Line 1 ===
        label = Text("Learning Rate", font_size=30, color=WHITE)
        self.place_at_grid(label, 'B3', scale_factor=0.6)
        self.play(self.lecture[0].animate.set_color(WHITE), Write(label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FF0000"))
        # Large step
        self.play(dot.animate.move_to(axes.c2p(2.5, 6.25)), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#00FF00"))
        # Small step
        dot.move_to(axes.c2p(-2.5, 6.25))
        self.play(dot.animate.move_to(axes.c2p(0, 0)), run_time=3)
        self.wait(1)
