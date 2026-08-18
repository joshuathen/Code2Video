from manim import *
import numpy as np

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
        self.setup_layout("Summary and Real-World Application", [
            "Conic sections appear in nature everywhere.",
            "From orbital paths to architectural acoustics.",
            "Dandelin spheres reveal their hidden logic."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Display a Whispering Gallery room layout using [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/gallery.svg]
        room = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/gallery.svg", color=WHITE)
        self.place_in_area(room, "A2", "D4", scale_factor=0.8)
        self.add(room)
        self.lecture[0].set_color("#FFFFFF")
        self.wait(2)

        # === Animation for Lecture Line 2 ===
        # Animate sound waves bouncing between two elliptical focal points
        f1 = Dot(point=room.get_center() + LEFT*1.2, color="#FF9900")
        f2 = Dot(point=room.get_center() + RIGHT*1.2, color="#FF9900")
        
        wave = Line(f1.get_center(), room.get_edge_center(UP), color="#FF9900")
        wave2 = Line(room.get_edge_center(UP), f2.get_center(), color="#FF9900")
        
        self.add(f1, f2, wave, wave2)
        self.lecture[1].set_color("#FF9900")
        self.play(Create(wave), Create(wave2))
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        # Fade out to show orbital mechanics planetary paths using [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/planet.svg]
        self.play(FadeOut(room), FadeOut(wave), FadeOut(wave2), FadeOut(f1), FadeOut(f2))
        
        orbit = Ellipse(width=3.0, height=1.5, color="#3399FF")
        planet = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/planet.svg", color="#3399FF")
        sun = Dot(color="#FFD700", radius=0.2)
        
        self.place_in_area(orbit, "B3", "E6", scale_factor=0.9)
        self.place_at_grid(sun, "C4", scale_factor=0.7)
        
        # Planet needs to follow the orbit path
        planet.add_updater(lambda m: m.move_to(orbit.point_from_proportion((self.time*0.5)%1)))
        
        self.add(orbit, planet, sun)
        self.lecture[2].set_color("#3399FF")
        self.wait(4)
