from manim import *
import os

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
        self.setup_layout("Generalizing to Parabola and Hyperbola", [
            "Parabolas need just one single sphere.", 
            "Hyperbolas require spheres in both nappes.", 
            "The plane cuts through the entire cone."
        ])
        
        # Load assets
        sphere_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/sphere.svg")
        cone_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/cone.svg")

        sphere1 = sphere_asset.copy().set_color("#3399FF")
        sphere2 = sphere_asset.copy().set_color("#3399FF")
        sphere3 = sphere_asset.copy().set_color("#3399FF")
        cone_repr = cone_asset.copy().set_color(GRAY)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#3399FF")
        # Parabola: One sphere
        self.place_at_grid(sphere1, 'C4', scale_factor=0.5)
        self.play(FadeIn(sphere1))

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#3399FF")
        # Hyperbola: Two spheres
        self.place_at_grid(sphere2, 'C5', scale_factor=0.5)
        self.place_at_grid(sphere3, 'D5', scale_factor=0.5)
        self.place_at_grid(cone_repr, 'C5', scale_factor=0.5)
        self.play(FadeIn(sphere2), FadeIn(sphere3), FadeIn(cone_repr))

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FF0000")
        # Highlight tangency points
        tangency1 = Dot(color="#FF0000").scale(0.5)
        tangency2 = Dot(color="#FF0000").scale(0.5)
        
        # Position using grid as per advice, tethered visually
        self.place_at_grid(tangency1, 'B4', scale_factor=0.5)
        self.place_at_grid(tangency2, 'B5', scale_factor=0.5)
        
        self.play(FadeIn(tangency1), FadeIn(tangency2))
        self.wait(2)
