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
            "Hyperspheres barely touch hypercube walls.",
            "Corners of the cube become increasingly vast.",
            "Points in high dimensions are very sparse."
        ]
        self.setup_layout("The Corner Concentration Mystery (2:45-4:15)", lecture_lines)
        
        # Load assets
        sphere_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/sphere.svg")
        cube_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/cube.svg")
        
        # Initialize
        touch_animation = VGroup(cube_asset, sphere_asset)
        corners_animation = cube_asset.copy()
        sparsity_animation = cube_asset.copy()
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFFFFF")
        self.place_in_area(touch_animation, 'A3', 'C6', scale_factor=0.6)
        self.play(FadeIn(touch_animation))

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FFD700")
        self.place_in_area(corners_animation, 'D1', 'F2', scale_factor=0.5)
        # Using simple geometric representation for corners as per original
        corners_lines = VGroup(*[Line(corners_animation.get_center(), corners_animation.get_corner(c), color="#FFD700") for c in [UL, UR, DL, DR]])
        self.play(FadeIn(corners_animation), Create(corners_lines))

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#00FFFF")
        self.place_in_area(sparsity_animation, 'D4', 'F6', scale_factor=0.5)
        dots = VGroup(*[Dot(point=sparsity_animation.get_center() + np.array([np.random.uniform(-0.4, 0.4), np.random.uniform(-0.4, 0.4), 0]), color="#00FFFF", radius=0.03) for _ in range(20)])
        self.play(FadeIn(sparsity_animation), FadeIn(dots))
        self.wait(2)
