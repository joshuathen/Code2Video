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
        self.setup_layout("Conclusion and Real-World Scaling", ["Binary is the puzzle's blueprint.", "Exponential growth limits scale.", "Large towers require immense time."])
        
        # Asset path
        tower_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/tower.svg"
        
        # Animations
        # === Animation for Lecture Line 1 ===
        # Show graph of moves growing exponentially, representing the tower growth.
        axes = Axes(x_range=[0, 10, 2], y_range=[0, 1024, 256], axis_config={"include_tip": False})
        graph = axes.plot(lambda x: 2**x - 1, color=YELLOW)
        tower_icon = SVGMobject(tower_path)
        group = VGroup(axes, graph, tower_icon)
        self.place_in_area(group, 'B1', 'E6', scale_factor=0.45)
        self.play(Create(group))
        self.lecture[0].set_color(YELLOW)

        # === Animation for Lecture Line 2 ===
        # Display formula 2^n - 1 in cyan (#00FFFF).
        formula = MathTex(r"2^n - 1", color="#00FFFF")
        self.place_at_grid(formula, 'E3', scale_factor=1.0)
        self.play(Write(formula))
        self.lecture[1].set_color("#00FFFF")

        # === Animation for Lecture Line 3 ===
        # Scale graph to illustrate massive real-world scale compared to the tower.
        self.play(group.animate.scale(1.1), run_time=2)
        self.lecture[2].set_color(RED)
        self.wait(2)
