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
            "Visualize the sphere in dimensions beyond three.",
            "A hypersphere follows the equation sum x squared.",
            "Slices reveal circles in two dimensions.",
            "A 3D sphere slicing yields a 2D circle.",
            "Expanding this logic leads to 4D geometry."
        ]
        self.setup_layout("Introduction: Beyond 3D Intuition", lecture_lines)
        
        # Load Assets
        sphere_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/sphere.svg")
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFFFFF"))
        self.place_in_area(sphere_asset.copy(), "A1", "C6", scale_factor=0.8)
        self.play(FadeIn(sphere_asset))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FFD700"))
        equation = MathTex(r"\sum x_i^2 = R^2", color="#FFD700")
        self.place_in_area(equation, 'A4', 'C6', scale_factor=0.9)
        self.play(Write(equation))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#00FFFF"))
        slice_circle = Circle(radius=0.5, color="#00FFFF")
        sphere_3d = sphere_asset.copy()
        self.place_at_grid(sphere_3d, 'A2', scale_factor=0.7)
        self.place_at_grid(slice_circle, 'D4', scale_factor=0.8)
        self.play(FadeIn(sphere_3d), Create(slice_circle))

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color("#FF69B4"))
        self.play(Indicate(sphere_3d), run_time=1)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color("#32CD32"))
        sphere_final = sphere_asset.copy()
        self.place_at_grid(sphere_final, 'C3', scale_factor=1.0)
        self.play(FadeIn(sphere_final))
        self.wait(1)
