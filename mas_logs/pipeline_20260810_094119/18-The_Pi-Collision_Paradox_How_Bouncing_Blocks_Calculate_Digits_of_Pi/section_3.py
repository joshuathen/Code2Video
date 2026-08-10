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
        self.setup_layout("The Geometry of Phase Space", [
            "Plot block positions as XY coordinates.",
            "Every collision is a boundary bounce.",
            "This traces a 2D zigzag path."
        ])
        
        # Phase Space Grid
        axes = Axes(
            x_range=[0, 6, 1], y_range=[0, 6, 1], 
            axis_config={"include_tip": False, "color": "#808080"}
        )
        
        # Integrating Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/block.svg
        block_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/block.svg")
        # Placing the icon as requested in storyboard
        self.place_in_area(block_icon, "A1", "A6", scale_factor=0.3)
        
        # Implementing Fix 28 for axes placement based on feedback
        self.place_in_area(axes, "C2", "F5", scale_factor=0.65)
        
        # Trajectory
        path = VMobject(color="#00FF00")
        path.set_points_smoothly([axes.c2p(0, 0), axes.c2p(2, 4), axes.c2p(4, 2), axes.c2p(6, 6)])
        
        dot = Dot(color="#00FF00")
        dot.move_to(axes.c2p(0, 0))
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#00FF00"))
        self.play(Create(axes), FadeIn(block_icon))
        
        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#00FF00"))
        self.play(FadeIn(dot))
        
        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#00FF00"))
        self.play(Create(path), run_time=2)
        self.play(MoveAlongPath(dot, path), run_time=3)
        self.play(Flash(dot, color="#FFFFFF"))
