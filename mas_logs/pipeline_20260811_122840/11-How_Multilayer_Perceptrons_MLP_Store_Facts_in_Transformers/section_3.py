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
        self.setup_layout("Mechanism: The Key-Value Memory Model", [
            "Input triggers a specific neuron.",
            "Activation functions act as switches.",
            "Output retrieves the stored 'Value' vector."
        ])
        
        # Create visual elements
        # Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/switch.svg
        try:
            input_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/switch.svg")
        except:
            input_icon = Square(color=WHITE)
            
        slot = Square(side_length=1.0, color=WHITE)
        value_vector = Arrow(ORIGIN, RIGHT, color="#9B59B6")

        # === Animation for Lecture Line 1 ===
        # "Input triggers a specific neuron."
        self.lecture[0].set_color(YELLOW)
        self.place_at_grid(slot, 'C3')
        self.play(FadeIn(slot))
        self.place_at_grid(input_icon, 'A3')
        self.play(input_icon.animate.move_to(slot.get_center()))

        # === Animation for Lecture Line 2 ===
        # "Activation functions act as switches."
        self.lecture[1].set_color("#2ECC71")
        self.play(slot.animate.set_color("#2ECC71"))

        # === Animation for Lecture Line 3 ===
        # "Output retrieves the stored 'Value' vector."
        self.lecture[2].set_color("#9B59B6")
        self.play(FadeOut(input_icon), FadeOut(slot))
        self.place_at_grid(value_vector, 'C3')
        self.play(FadeIn(value_vector))
        self.wait(1)
