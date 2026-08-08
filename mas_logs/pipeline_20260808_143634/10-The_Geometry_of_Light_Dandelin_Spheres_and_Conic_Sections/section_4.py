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
        self.setup_layout("Connecting Shapes to Spheres", [
            "Adding a second sphere generalizes the proof.",
            "Parabolas use one sphere in the cone.",
            "Hyperbolas use spheres in different cone nappes."
        ])
        
        # Load asset
        sphere_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/sphere.svg"
        
        # === Animation for Lecture Line 1 ===
        # Adding a second sphere generalizes the proof.
        sphere1 = SVGMobject(sphere_path, color="#FF8000")
        sphere2 = SVGMobject(sphere_path, color="#FF8000")
        self.place_at_grid(sphere1, 'B2', scale_factor=0.5)
        self.place_at_grid(sphere2, 'B4', scale_factor=0.5)
        self.play(FadeIn(sphere1), FadeIn(sphere2))
        self.lecture[0].set_color("#FF8000")
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Parabolas use one sphere in the cone.
        sphere3 = SVGMobject(sphere_path, color="#00FF00")
        self.place_at_grid(sphere3, 'E3', scale_factor=0.5)
        self.play(FadeOut(sphere1), FadeOut(sphere2), FadeIn(sphere3))
        self.lecture[1].set_color("#00FF00")
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Hyperbolas use spheres in different cone nappes.
        sphere4 = SVGMobject(sphere_path, color="#FF0000")
        sphere5 = SVGMobject(sphere_path, color="#FF0000")
        self.place_at_grid(sphere4, 'D2', scale_factor=0.5)
        self.place_at_grid(sphere5, 'E5', scale_factor=0.5)
        self.play(FadeOut(sphere3), FadeIn(sphere4), FadeIn(sphere5))
        self.lecture[2].set_color("#FF0000")
        self.wait(1)
