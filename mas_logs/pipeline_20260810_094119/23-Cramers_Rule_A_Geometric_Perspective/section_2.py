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

class Section2Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "We solve the system: x v1 + y v2 = b.",
            "Visualize v1 and v2 as grid builders.",
            "Aiming to reach the target vector b."
        ]
        self.setup_layout("The Geometric Setup: Solving Ax = b", lecture_lines)
        
        # Assets (Using provided asset paths)
        # Assuming SVG icons are small icons; we'll treat them as markers
        grid_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/grid.svg")
        vector_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/vector.svg")
        target_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/target.svg")

        # Initialize core geometry
        axes = Axes(x_range=[-1, 4, 1], y_range=[-1, 4, 1], axis_config={"include_tip": True}).scale(0.5)
        v1 = Vector([2, 1], color=BLUE)
        v2 = Vector([1, 2], color=YELLOW)
        b = Vector([3, 3], color=RED)
        
        # VGroup for scene
        v_group = VGroup(axes, v1, v2, b, grid_icon, vector_icon, target_icon)
        
        # Applying layout constraints (Issue 35/24)
        self.place_in_area(v_group, 'B3', 'F5', scale_factor=0.8)
        self.add(v_group)
        
        # Initial hiding for reveal animations
        grid_icon.set_opacity(0)
        vector_icon.set_opacity(0)
        target_icon.set_opacity(0)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(BLUE))
        self.play(FadeIn(vector_icon))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(YELLOW))
        self.play(FadeIn(grid_icon))
        parallelogram = Polygon(ORIGIN, v1.get_end(), v1.get_end() + v2.get_end(), v2.get_end(), color=WHITE, fill_opacity=0.2)
        self.play(Create(parallelogram))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(RED))
        self.play(FadeIn(target_icon))
        self.play(Indicate(b))
        self.wait(2)
