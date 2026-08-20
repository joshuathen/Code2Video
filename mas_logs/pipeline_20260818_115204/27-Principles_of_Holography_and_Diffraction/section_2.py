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
        self.setup_layout("The Mechanism of Diffraction", [
            "Diffraction occurs when waves encounter apertures.",
            "Huygens-Fresnel explains spreading wave behavior.",
            "Gratings map spatial information through diffraction."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Represent aperture and waves
        slit = Line(UP*0.5, DOWN*0.5, color=GREY).set_stroke(width=8)
        self.place_at_grid(slit, 'D2', scale_factor=0.7)
        
        wave_animation = VGroup(*[Arc(radius=0.2+i*0.2, angle=PI/2, start_angle=-PI/4, color=BLUE) for i in range(5)])
        self.place_at_grid(wave_animation, 'B4', scale_factor=0.6)
        
        self.play(FadeIn(slit))
        self.play(self.lecture[0].animate.set_color(BLUE))
        self.play(Create(wave_animation))
        self.play(wave_animation.animate.scale(2).shift(RIGHT*1.5))

        # === Animation for Lecture Line 2 ===
        # Huygens Principle (Multiple source points)
        self.play(self.lecture[1].animate.set_color(YELLOW))
        sources = VGroup(*[Dot(color=YELLOW).move_to(slit.get_center() + UP*i*0.2) for i in [-2, -1, 0, 1, 2]])
        self.play(FadeIn(sources))
        
        wavefronts = VGroup(*[Circle(radius=0.5, color=YELLOW, stroke_opacity=0.5).move_to(s.get_center()) for s in sources])
        self.play(Create(wavefronts))
        self.play(wavefronts.animate.scale(2))

        # === Animation for Lecture Line 3 ===
        # Grating representation
        self.play(self.lecture[2].animate.set_color(GREEN))
        grating = VGroup(*[Line(UP*0.5, DOWN*0.5, color=WHITE).shift(RIGHT*i*0.2) for i in range(-2, 3)])
        self.place_at_grid(grating, 'D5', scale_factor=0.7)
        
        screen = Line(UP*1.5, DOWN*1.5, color=GREY).shift(RIGHT*3)
        self.play(FadeIn(grating), FadeIn(screen))
        
        dots = VGroup(*[Dot(color=GREEN).move_to(screen.get_center() + UP*i*0.4) for i in [-2, -1, 0, 1, 2]])
        self.play(FadeIn(dots))
        self.wait(1)
