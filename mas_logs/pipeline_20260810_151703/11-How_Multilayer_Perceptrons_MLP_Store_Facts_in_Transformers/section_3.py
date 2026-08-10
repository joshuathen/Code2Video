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
        self.setup_layout("The Mechanism: Nonlinear Activation as a Switch", [
            "Nonlinear activation functions act as selective switches.",
            "They fire only when input matches specific patterns.",
            "This ensures sparse and modular factual storage."
        ])
        
        # --- Assets ---
        neuron_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/neuron.svg")
        switch_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/switch.svg")
        
        # 1. Neuron Gate Mobject
        gate_label = Text("Gate", font_size=24, color="#FF5733")
        neuron_group = VGroup(neuron_icon, gate_label).arrange(DOWN)
        # Requirement: Line 58: self.place_in_area(neuron_group, 'C2', 'C4', scale_factor=0.9)
        self.place_in_area(neuron_group, 'C2', 'C4', scale_factor=0.9)

        # 2. ReLU Function Mobject
        axes = Axes(x_range=[-2, 2, 1], y_range=[-1, 2, 1], axis_config={"include_tip": False}, x_length=3, y_length=2)
        relu_curve = axes.plot(lambda x: max(0, x), color="#33FF57")
        relu_group = VGroup(axes, relu_curve)
        # Requirement: Line 64: self.place_in_area(relu_group, 'E3', 'F5', scale_factor=1.0)
        self.place_in_area(relu_group, 'E3', 'F5', scale_factor=1.0)

        # 3. Switch icon for flashing
        switch_icon.set_color("#3357FF")
        self.place_at_grid(switch_icon, 'B5', scale_factor=0.8)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FF5733")
        self.play(FadeIn(neuron_group))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#33FF57")
        self.play(Create(relu_group))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#3357FF")
        # Flash animation with switch icon
        self.play(Flash(switch_icon, color="#3357FF", num_lines=15), run_time=1.5)
        self.wait(1)
