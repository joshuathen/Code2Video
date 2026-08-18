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

class Section1Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "Tangent lines approximate curves at a point.",
            "They predict where the curve crosses the axis.",
            "This allows us to estimate roots numerically."
        ]
        self.setup_layout("Prerequisites: The Geometry of a Tangent", lecture_lines)
        
        # Create function and axes
        axes = Axes(x_range=[-1, 5], y_range=[-1, 4], axis_config={"include_tip": False})
        curve = axes.plot(lambda x: 0.2*(x-1)*(x-3)*(x-4) + 2, color=WHITE)
        graph = VGroup(axes, curve)
        
        # Fix: axes and graph overflow: use self.place_in_area(axes, 'C2', 'F6', scale_factor=0.6)
        self.place_in_area(graph, "C2", "F6", scale_factor=0.6)
        self.add(graph)
        
        # Elements for animation
        x0_val = 1.2
        f_x0 = 0.2*(x0_val-1)*(x0_val-3)*(x0_val-4) + 2
        
        # Point and Tangent
        x0_point = Dot(axes.c2p(x0_val, f_x0), color=YELLOW)
        
        # Fix: Tangent point centering
        self.place_at_grid(x0_point, "D3", scale_factor=0.7)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))
        self.play(Create(x0_point))
        
        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(ORANGE))
        
        # Slope (derivative approx)
        df_x0 = 0.6*x0_val**2 - 4*x0_val + 6.2
        
        tangent = Line(
            axes.c2p(x0_val-1, f_x0 - df_x0),
            axes.c2p(x0_val+1, f_x0 + df_x0),
            color=PURPLE
        )
        self.play(Create(tangent))
        
        # Intersection x1
        x1_val = x0_val - f_x0 / df_x0
        x1_point = Dot(axes.c2p(x1_val, 0), color=RED)
        self.play(Create(x1_point))
        
        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(RED))
        
        label_x0 = MathTex("x_0")
        label_x1 = MathTex("x_1")
        
        # Fix: Label positioning
        self.place_at_grid(label_x0, "D4", scale_factor=0.7)
        self.place_at_grid(label_x1, "F2", scale_factor=0.7)
        
        # Placeholder for asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg
        # Note: '/scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg' isn't a valid visual asset, using a simple marker instead if needed.
        # Given instruction: "Load and place the referenced files... while preserving layout constraints"
        # Since it is a '/scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg', I will skip adding it as an actual file load to avoid errors.
        
        self.play(Write(label_x0), Write(label_x1))
        
        self.wait(2)
