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

class Section3Scene(TeachingScene):
    def construct(self):
        self.setup_layout("The Measurement Problem", [
            "Observation causes immediate wavefunction collapse.",
            "Schrödinger’s cat illustrates this quantum paradox.",
            "Measurement reveals zero with probability |α|²."
        ])
        
        # Mobjects for animations
        # Asset usage: /scratch/pawsey1357/jthen/Code2Video/assets/icon/box.svg
        box = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/box.svg", color=WHITE)
        self.place_in_area(box, 'B2', 'D4', scale_factor=0.9)
        
        state_text = MathTex(r"|\\text{Alive}\\rangle + |\\text{Dead}\\rangle", font_size=30, color=WHITE)
        self.place_at_grid(state_text, 'C3', scale_factor=0.7) # Apply B020
        
        operator = MathTex(r"\\hat{M}", font_size=40, color="#FF4500")
        self.place_at_grid(operator, 'D4', scale_factor=0.8) # Resolution for ID 33/48

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#00FFFF"))
        self.play(FadeIn(box), Write(state_text))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FF4500"))
        self.play(operator.animate.move_to(self.grid['C4']))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FF00FF"))
        # Opening the box: collapse
        self.play(FadeOut(state_text), FadeOut(operator))
        
        # Asset usage: /scratch/pawsey1357/jthen/Code2Video/assets/icon/cat.png
        cat = ImageMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/cat.png")
        self.place_at_grid(cat, 'C3', scale_factor=0.5)
        
        final_state = MathTex(r"|0\\rangle", font_size=40, color="#FF00FF")
        self.place_at_grid(final_state, 'D2', scale_factor=1.0)
        self.play(FadeIn(cat), Write(final_state))
        self.wait(2)
