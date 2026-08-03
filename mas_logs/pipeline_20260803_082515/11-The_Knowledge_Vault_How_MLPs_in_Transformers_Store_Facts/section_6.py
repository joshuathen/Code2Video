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

class Section6Scene(TeachingScene):
    def construct(self):
        # Data from storyboard
        title = "Summary: The Key-Value Memory Store"
        lecture_lines = [
            "MLPs operate as a programmable associative memory system.",
            "Key vectors detect patterns while value vectors store facts.",
            "The activation gate controls the flow between them.",
            "Millions of these pairs form the model's global memory.",
            "Together, they transform input queries into factual knowledge."
        ]
        
        # Initialize Layout
        self.setup_layout(title, lecture_lines)

        # Highlighting colors
        c_system = "#88C0D0"  # Light Blue
        c_kv = "#EBCB8B"      # Yellow/Gold
        c_gate = "#A3BE8C"    # Green
        c_memory = "#BF616A"  # Red/Coral
        c_final = "#FFFFFF"   # White

        # Assets
        key_icon_w1 = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/key.svg").scale(0.3).set_color(c_system)
        lock_icon_gelu = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/lock.svg").scale(0.3).set_color(WHITE)
        key_icon_summary = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/key.svg").scale(0.4).set_color(c_final)

        # === Animation for Lecture Line 1 ===
        # "MLPs operate as a programmable associative memory system."
        self.play(self.lecture[0].animate.set_color(c_system))
        
        # Create blocks
        w1_box = Rectangle(width=1.2, height=1.2, color=c_system)
        w1_text = Text("W1\n(Key Match)", font_size=16, color=c_system)
        w1_content = VGroup(w1_text, key_icon_w1).arrange(DOWN, buff=0.1)
        w1_group = VGroup(w1_box, w1_content) 
        self.place_in_area(w1_group, "B1", "C2")

        gelu_box = Rectangle(width=1.2, height=1.2, color=WHITE)
        gelu_text = Text("GELU\nGate", font_size=16, color=WHITE)
        gelu_content = VGroup(gelu_text, lock_icon_gelu).arrange(DOWN, buff=0.1)
        gelu_group = VGroup(gelu_box, gelu_content)
        self.place_in_area(gelu_group, "B3", "C4")

        w2_box = Rectangle(width=1.2, height=1.2, color=WHITE)
        w2_text = Text("W2\n(Retrieve Value)", font_size=16, color=WHITE)
        w2_group = VGroup(w2_box, w2_text)
        self.place_in_area(w2_group, "B5", "C6", scale_factor=0.9) # Issue 43/44 fix

        # Connection arrows
        arrow1 = Arrow(w1_box.get_right(), gelu_box.get_left(), buff=0.1, color=WHITE)
        arrow2 = Arrow(gelu_box.get_right(), w2_box.get_left(), buff=0.1, color=WHITE)
        
        self.play(
            Create(w1_group),
            Create(arrow1),
            Create(gelu_group),
            Create(arrow2),
            Create(w2_group)
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # "Key vectors detect patterns while value vectors store facts."
        self.play(
            self.lecture[1].animate.set_color(c_kv),
            w1_box.animate.set_color(c_kv),
            w1_text.animate.set_color(c_kv),
            key_icon_w1.animate.set_color(c_kv),
            w2_box.animate.set_color(c_kv),
            w2_text.animate.set_color(c_kv)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # "The activation gate controls the flow between them."
        self.play(
            self.lecture[2].animate.set_color(c_gate),
            gelu_box.animate.set_color(c_gate),
            gelu_text.animate.set_color(c_gate),
            lock_icon_gelu.animate.set_color(c_gate)
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # "Millions of these pairs form the model's global memory."
        self.play(self.lecture[3].animate.set_color(c_memory))
        
        # Pulse traveling from left to right
        pulse = Dot(color=c_memory, radius=0.1)
        pulse.move_to(w1_box.get_left())
        
        self.add(pulse)
        self.play(
            pulse.animate.move_to(w2_box.get_right()),
            run_time=2,
            rate_func=linear
        )
        self.play(FadeOut(pulse))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # "Together, they transform input queries into factual knowledge."
        self.play(self.lecture[4].animate.set_color(c_final))
        
        summary_text = Text("A Massive Key-Value Memory Store", font_size=24, color=c_final)
        summary_group = VGroup(summary_text, key_icon_summary).arrange(DOWN, buff=0.2)
        
        # Position summary text at the specified area of the grid (Issue 42 fix)
        self.place_in_area(summary_group, "E2", "F5", scale_factor=0.9)
        
        self.play(Write(summary_group))
        self.wait(3)
