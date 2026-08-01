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

class Section1Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            'In our daily life, objects have definite states.', 
            'A light switch is either up or down.', 
            'But atoms follow a different set of rules.', 
            'Quantum objects can exist in multiple states simultaneously.', 
            'This strange "both/and" phenomenon is called superposition.'
        ]
        self.setup_layout("The Intuition Gap: Classical vs. Quantum", lecture_lines)
        
        # Asset Path
        switch_path = "/mmfs1/data/home/jthen/Code2Video/assets/icon/switch.svg"

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        
        classical_label = Text("Classical", color="#FFFFFF", font_size=24)
        self.place_at_grid(classical_label, "A2")
        
        # Classical switch in OFF state (represented by 180 deg rotation)
        classical_switch = SVGMobject(switch_path, color="#FFFFFF")
        self.place_in_area(classical_switch, "B1", "E3", scale_factor=1.5)
        classical_switch.rotate(PI) 
        
        off_label = Text("OFF", font_size=18, color="#FFFFFF")
        off_label.next_to(classical_switch, DOWN, buff=0.2)
        
        self.play(FadeIn(classical_label), FadeIn(classical_switch), FadeIn(off_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        on_label = Text("ON", font_size=18, color="#FFFFFF")
        on_label.next_to(classical_switch, UP, buff=0.2)
        
        # Toggling logic
        for _ in range(2):
            self.play(
                classical_switch.animate.rotate(PI),
                FadeOut(off_label),
                FadeIn(on_label),
                run_time=0.5
            )
            self.play(
                classical_switch.animate.rotate(PI),
                FadeOut(on_label),
                FadeIn(off_label),
                run_time=0.5
            )
        
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        quantum_label = Text("Quantum", color="#00FFFF", font_size=24)
        self.place_at_grid(quantum_label, "A5")
        
        # Glowing Sphere
        quantum_sphere = Circle(radius=0.9, color="#00FFFF", fill_opacity=0.3)
        glow_layer = Circle(radius=1.2, color="#00FFFF", fill_opacity=0.1, stroke_width=0)
        self.place_in_area(quantum_sphere, "B4", "E6")
        self.place_in_area(glow_layer, "B4", "E6")
        
        self.play(
            FadeIn(quantum_label),
            FadeIn(quantum_sphere),
            FadeIn(glow_layer)
        )
        
        # Pulsating cyan energy
        self.play(
            quantum_sphere.animate.scale(1.1).set_opacity(0.5),
            glow_layer.animate.scale(1.2).set_opacity(0.05),
            rate_func=there_and_back,
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)
        
        # Translucent switches representing UP and DOWN
        s_up = SVGMobject(switch_path, color="#FF00FF", fill_opacity=0.5)
        self.place_at_grid(s_up, "B5", scale_factor=0.7)
        
        s_down = SVGMobject(switch_path, color="#FF00FF", fill_opacity=0.5)
        self.place_at_grid(s_down, "E5", scale_factor=0.7)
        s_down.rotate(PI) # Represent DOWN
        
        label_up = Text("UP", font_size=16, color="#FF00FF")
        label_up.next_to(s_up, UP, buff=0.1)
        
        label_down = Text("DOWN", font_size=16, color="#FF00FF")
        label_down.next_to(s_down, DOWN, buff=0.1)
        
        self.play(
            FadeIn(s_up), 
            FadeIn(s_down), 
            FadeIn(label_up), 
            FadeIn(label_down),
            quantum_sphere.animate.set_opacity(0.1)
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)
        
        superposition_text = Text("SUPERPOSITION", color="#00FFFF", font_size=22)
        # Place above sphere area (replacing or near Quantum label)
        self.place_at_grid(superposition_text, "A5")
        
        # Hide quantum label to make room for flash
        self.remove(quantum_label)
        
        # Flashing and rapid pulse
        for _ in range(3):
            self.play(
                superposition_text.animate.set_opacity(1),
                quantum_sphere.animate.scale(1.2).set_color("#00FFFF"),
                run_time=0.3
            )
            self.play(
                superposition_text.animate.set_opacity(0.3),
                quantum_sphere.animate.scale(0.833).set_color("#00FFFF"),
                run_time=0.3
            )
        
        self.play(superposition_text.animate.set_opacity(1))
        self.wait(3)
