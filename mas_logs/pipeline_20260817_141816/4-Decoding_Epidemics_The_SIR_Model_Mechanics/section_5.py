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

class Section5Scene(TeachingScene):
    def construct(self):
        lecture_lines = ["Models simplify complex biological realities.", "Vaccines remove people from Susceptible.", "These tools ultimately save lives."]
        self.setup_layout("Conclusion: From Models to Reality", lecture_lines)
        
        # Elements for the right side
        sir_label = Text("SIR Model", font_size=24, color=BLUE)
        vaccine_label = Text("Vaccinated", font_size=24, color=GREEN)
        
        # Load assets
        vaccine_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/vaccine.svg")
        hospital_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/hospital.svg")
        
        # Placeholder curves
        sir_curve = ParametricFunction(lambda t: np.array([t, 0.5*np.sin(t*2), 0]), t_range=[0, 3], color=BLUE)
        vaccine_curve = ParametricFunction(lambda t: np.array([t, 0.2*np.sin(t*2), 0]), t_range=[0, 3], color=GREEN)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(BLUE))
        self.place_at_grid(sir_label, 'A2', scale_factor=0.8)
        self.place_in_area(sir_curve, 'B1', 'D3', scale_factor=0.9)
        self.place_at_grid(vaccine_icon, 'B3', scale_factor=0.5)
        self.play(Write(sir_label), Create(sir_curve), FadeIn(vaccine_icon))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(GREEN))
        self.place_at_grid(vaccine_label, 'A5', scale_factor=0.8)
        self.place_in_area(vaccine_curve, 'B4', 'D6', scale_factor=0.9)
        self.play(Write(vaccine_label), Create(vaccine_curve))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(YELLOW))
        final_message = Text("Predicting outcomes saves lives.", font_size=32, color=YELLOW)
        self.place_in_area(final_message, 'E1', 'F6', scale_factor=0.7)
        self.place_at_grid(hospital_icon, 'E6', scale_factor=0.6)
        self.play(FadeIn(final_message), FadeIn(hospital_icon))
        self.wait(2)
