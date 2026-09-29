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

class Section1Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "Refractive index defines how light slows down.",
            "We model electrons as masses on springs.",
            "Atoms have a natural vibration frequency."
        ]
        self.setup_layout("Hook & Prerequisite Review", lecture_lines)
        
        # Assets
        atom_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/atom.svg"
        spring_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/spring.svg"
        
        # Initialize animation objects
        title = Text("Light Interaction", font_size=36, color=WHITE)
        self.place_in_area(title, 'D2', 'D5', scale_factor=0.9)
        
        ray = Line(start=np.array([-1, 0, 0]), end=np.array([1, 0, 0]), color=YELLOW)
        ray_label = Text("Ray", font_size=24, color="#FF00FF")
        self.place_at_grid(ray_label, 'D1', scale_factor=0.7)
        
        atom_img = SVGMobject(atom_path) if os.path.exists(atom_path) else Circle(radius=0.3, color=BLUE)
        spring_img = SVGMobject(spring_path) if os.path.exists(spring_path) else Rectangle(width=0.4, height=0.6, color=RED)
        
        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(title))
        self.play(self.lecture[0].animate.set_color(YELLOW))
        
        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#00FF00"))
        # Fade in ray with atom
        self.place_at_grid(atom_img, 'B3', scale_factor=0.5)
        self.place_at_grid(ray, 'B4', scale_factor=0.8)
        self.play(FadeIn(atom_img), Create(ray))
        
        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#00FFFF"))
        self.play(title.animate.set_color("#00FFFF"))
        # Accompanied by spring
        self.place_at_grid(spring_img, 'E3', scale_factor=0.5)
        self.play(FadeIn(spring_img))
        
        self.wait(2)
