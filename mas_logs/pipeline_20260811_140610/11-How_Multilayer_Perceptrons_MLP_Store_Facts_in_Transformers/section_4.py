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
        self.setup_layout("Case Study: Fact Editing and Activation", [
            "Specific neurons encode individual factual knowledge.",
            "Training adjusts weights to update associations.",
            "We can suppress or excite specific facts."
        ])
        
        self.lecture.set_opacity(0)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_opacity(1)
        self.lecture[0].set_color("#3498DB")
        
        # Using SVG Asset
        neuron_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/neuron.svg", color="#3498DB")
        self.place_at_grid(neuron_icon, "C5", scale_factor=0.6)
        
        fact_neuron_label = Text("Fact Neuron", font_size=20, color="#3498DB")
        # B011: tethered
        fact_neuron_label.next_to(neuron_icon, UP, buff=0.1)
        
        self.play(FadeIn(neuron_icon), Write(fact_neuron_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_opacity(1)
        self.lecture[1].set_color("#E74C3C")
        
        # Weight mod
        weight_conn = Line(start=neuron_icon.get_right(), end=neuron_icon.get_right() + RIGHT*1.0, color=WHITE)
        weight_val = Text("w", font_size=24, color=WHITE)
        weight_val.next_to(weight_conn, UP, buff=0.1)
        
        self.play(Create(weight_conn), Write(weight_val))
        self.play(weight_conn.animate.set_color("#E74C3C"), weight_val.animate.set_color("#E74C3C"))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_opacity(1)
        self.lecture[2].set_color("#2ECC71")
        
        # Asset: final neuron update
        neuron_icon_2 = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/neuron.svg", color="#2ECC71")
        self.place_at_grid(neuron_icon_2, "E5", scale_factor=0.6)
        
        update_label = Text("Update", font_size=20, color="#2ECC71")
        update_label.next_to(neuron_icon_2, UP, buff=0.1)
        
        self.play(FadeIn(neuron_icon_2), Write(update_label))
        self.play(neuron_icon.animate.set_color("#2ECC71"), fact_neuron_label.animate.set_color("#2ECC71"))
        self.wait(2)
