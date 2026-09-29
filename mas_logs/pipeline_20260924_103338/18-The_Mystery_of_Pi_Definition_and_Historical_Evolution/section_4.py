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
        self.setup_layout("Modern Perspectives and Wrap-up", ["Pi powers modern technology today.", "It helps your phone navigate routes.", "Pi bridges geometry and the infinite."])
        
        # Assets
        satellite = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/satellite.svg")
        phone = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/smartphone.svg")
        
        # === Animation for Lecture Line 1 ===
        # Pi powers modern technology today.
        self.lecture[0].set_color(YELLOW)
        
        self.place_at_grid(satellite, 'A2', scale_factor=0.4)
        gps_label = Text("GPS", font_size=20, color=BLUE).next_to(satellite, DOWN, buff=0.1)
        self.play(FadeIn(satellite), Write(gps_label), run_time=1.5)
        self.wait(2)

        # === Animation for Lecture Line 2 ===
        # It helps your phone navigate routes.
        self.lecture[1].set_color(GREEN)
        
        self.place_at_grid(phone, 'E2', scale_factor=0.4)
        nav_label = Text("Nav", font_size=20, color=GREEN).next_to(phone, DOWN, buff=0.1)
        
        # Animate digits cascading
        digits = VGroup(*[Text("π", font_size=24, color=YELLOW) for _ in range(5)]).arrange(DOWN)
        self.place_at_grid(digits, 'C6', scale_factor=0.5)
        
        self.play(FadeIn(phone), Write(nav_label), FadeIn(digits, shift=DOWN), run_time=2)
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        # Pi bridges geometry and the infinite.
        self.lecture[2].set_color(PURPLE)
        
        # Fade the central devices and digits out to black, leaving only the label 'π' in the center.
        pi_label = MathTex(r"\pi", font_size=96, color=WHITE)
        
        self.play(
            FadeOut(self.lecture), 
            FadeOut(self.title), 
            FadeOut(satellite), 
            FadeOut(gps_label),
            FadeOut(phone), 
            FadeOut(nav_label), 
            FadeOut(digits),
            run_time=2
        )
        self.play(Write(pi_label), run_time=1)
        self.wait(2)
