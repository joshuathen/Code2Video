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

class Section5Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Synthesis & Summary", [
            "MLPs are distributed storage systems.", 
            "Activating neurons reconstructs stored facts.", 
            "This is neural memory in action."
        ])
        
        # Define Neurons
        neuron_asset = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/neuron.svg"
        neurons = VGroup()
        for pos in ["B2", "B4", "C3", "D2", "D4"]:
            n = SVGMobject(neuron_asset) if os.path.exists(neuron_asset) else Dot(color=BLUE)
            self.place_at_grid(n, pos, scale_factor=0.5)
            neurons.add(n)
        
        lines = VGroup(
            Line(neurons[0].get_center(), neurons[2].get_center(), color=GRAY),
            Line(neurons[1].get_center(), neurons[2].get_center(), color=GRAY),
            Line(neurons[2].get_center(), neurons[3].get_center(), color=GRAY),
            Line(neurons[2].get_center(), neurons[4].get_center(), color=GRAY)
        )

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        self.play(FadeIn(lines), FadeIn(neurons))

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        pulse_group = VGroup(neurons[0], neurons[1], neurons[3], neurons[4])
        self.play(*[n.animate.set_color(ORANGE) for n in pulse_group], run_time=1.5)
        self.play(*[n.animate.set_color(WHITE) for n in pulse_group], run_time=1.5)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        memory_text = Text("Neural Memory", color="#ECF0F1", font_size=36)
        self.place_in_area(memory_text, "E1", "E6")
        self.play(Write(memory_text))
        self.wait(2)
