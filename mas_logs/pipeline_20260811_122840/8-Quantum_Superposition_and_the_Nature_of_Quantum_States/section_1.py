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
        self.setup_layout("Prerequisite: From Bits to Qubits", [
            "Classical bits are either zero or one.",
            "Qubits exist as vectors in Hilbert space.",
            "Think of a spinning coin as superposition."
        ])
        
        # Assets
        switch_asset = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/switch.svg"
        coin_asset = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/coin.svg"

        # === Animation for Lecture Line 1 ===
        # Classical bits are either zero or one.
        switch = SVGMobject(switch_asset, color="#FFFFFF")
        self.place_at_grid(switch, 'D2', scale_factor=0.4)
        
        label_0 = Text("0", font_size=36, color="#FFFFFF").next_to(switch, LEFT)
        label_1 = Text("1", font_size=36, color="#FFFFFF").next_to(switch, RIGHT)
        
        self.add(switch, label_0, label_1)
        self.lecture[0].set_color("#FFFFFF")
        self.play(Indicate(switch), run_time=1)
        self.play(FadeOut(switch), FadeOut(label_0), FadeOut(label_1))

        # === Animation for Lecture Line 2 ===
        # Qubits exist as vectors in Hilbert space.
        coin = SVGMobject(coin_asset, color="#FFD700")
        self.place_at_grid(coin, 'D5', scale_factor=0.6)
        
        label_0_state = MathTex(r"|0\rangle", color="#00FFFF")
        label_1_state = MathTex(r"|1\rangle", color="#00FFFF")
        labels = VGroup(label_0_state, label_1_state).arrange(DOWN)
        self.place_in_area(labels, 'C4', 'E6', scale_factor=0.5)
        
        self.add(coin, labels)
        self.lecture[1].set_color("#FFD700")
        self.play(FadeIn(coin), Write(labels))

        # === Animation for Lecture Line 3 ===
        # Think of a spinning coin as superposition.
        blur = Circle(radius=0.4, color="#FF00FF", stroke_opacity=0.3).move_to(coin.get_center())
        
        vec_concept = Arrow(start=coin.get_center(), end=coin.get_center() + RIGHT*0.8 + UP*0.5, color="#FFFF00")
        
        self.lecture[2].set_color("#FF00FF")
        self.play(Create(blur), GrowArrow(vec_concept))
        self.wait(2)
