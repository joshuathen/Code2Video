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

class Section1Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "Classical bits exist as either 0 or 1.",
            "Quantum states live in a complex Hilbert space.",
            "Imagine a toggle switch versus a dimmer knob.",
            "The dimmer represents infinite states between ON and OFF.",
            "This defines the basis of quantum superposition."
        ]
        self.setup_layout("The Classical vs. Quantum Divide", lecture_lines)
        
        # Assets
        switch_svg = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/switch.svg", color=WHITE)
        knob_svg = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/knob.svg", color=WHITE)
        
        # Mobjects
        switch_0 = switch_svg.copy()
        switch_1 = switch_svg.copy()
        switches = VGroup(switch_0, switch_1).arrange(RIGHT, buff=0.5)
        
        sphere = Sphere(radius=1, color=BLUE, fill_opacity=0.5)
        dot = Dot(color=YELLOW)
        knob = knob_svg.copy()
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        self.place_in_area(switches, 'B2', 'B3', scale_factor=0.5)
        self.play(Create(switches))

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(BLUE)
        self.place_in_area(sphere, 'D2', 'D3', scale_factor=0.4)
        self.play(FadeOut(switches), GrowFromCenter(sphere))

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(GREEN)
        self.place_at_grid(knob, 'F4', scale_factor=0.6)
        self.play(FadeIn(knob), FadeIn(dot))
        self.place_at_grid(dot, 'F3', scale_factor=0.6)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color(GREEN)
        path = Arc(radius=0.3, start_angle=0, angle=PI/2)
        self.play(MoveAlongPath(dot, path), Rotate(knob, angle=PI/4))

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color(PURPLE)
        self.play(Indicate(sphere), Indicate(dot))
        self.wait(1)
