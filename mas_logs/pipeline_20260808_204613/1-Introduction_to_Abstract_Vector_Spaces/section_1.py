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
            "Vectors are more than just geometric arrows.",
            "Consider vectors as abstract objects in sets.",
            "These sets follow specific rules of engagement.",
            "Linearity defines the structure of vector spaces.",
            "Abstract spaces generalize our 2D/3D knowledge."
        ]
        self.setup_layout("From Concrete to Abstract", lecture_lines)
        
        # Prep objects
        axes = Axes(x_range=[-2, 2], y_range=[-2, 2], axis_config={"include_tip": True}).scale(0.5)
        vec = Vector([1.5, 1], color=WHITE)
        comp_lines = VGroup(
            DashedLine(vec.get_end(), [vec.get_end()[0], 0, 0], color="#FF8080"),
            DashedLine(vec.get_end(), [0, vec.get_end()[1], 0], color="#FF8080")
        )
        point = Dot(vec.get_end(), color="#80FFFF")
        # Updated to use asset as per Issue 13
        grid_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/grid.svg")
        label = MathTex(r"v = (x, y)", color="#FFFF80")

        # === Animation for Lecture Line 1 ===
        self.place_at_grid(VGroup(axes, vec), 'C3', scale_factor=0.8)
        self.play(FadeIn(axes), GrowArrow(vec), self.lecture[0].animate.set_color(WHITE))

        # === Animation for Lecture Line 2 ===
        self.place_at_grid(comp_lines, 'C3', scale_factor=0.8)
        self.play(Create(comp_lines), self.lecture[0].animate.set_color(GRAY), self.lecture[1].animate.set_color("#FF8080"))

        # === Animation for Lecture Line 3 ===
        new_vec = Vector([0.5, 1.5], color="#80FF80")
        self.play(ReplacementTransform(vec, new_vec), self.lecture[1].animate.set_color(GRAY), self.lecture[2].animate.set_color("#80FF80"))
        vec = new_vec

        # === Animation for Lecture Line 4 ===
        # Fixes 18 & 16: Adjust scale and grid positioning
        self.place_at_grid(grid_asset, 'C3', scale_factor=0.6)
        self.place_at_grid(point, 'C2', scale_factor=0.8)
        self.play(Create(grid_asset), FadeIn(point), self.lecture[2].animate.set_color(GRAY), self.lecture[3].animate.set_color("#80FFFF"))

        # === Animation for Lecture Line 5 ===
        # Fix 17: Adjust area positioning
        self.place_in_area(label, 'D3', 'E5', scale_factor=0.7)
        self.play(FadeIn(label), self.lecture[3].animate.set_color(GRAY), self.lecture[4].animate.set_color("#FFFF80"))
