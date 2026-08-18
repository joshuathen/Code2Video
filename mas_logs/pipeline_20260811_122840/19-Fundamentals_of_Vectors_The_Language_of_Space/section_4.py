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
        self.setup_layout("Scaling Vectors", ["Scalars scale the vector's length.", "Direction remains the same.", "Negative scalars reverse the direction."])
        
        # Assets
        asset_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg"
        icon = SVGMobject(asset_path).scale(0.5)

        # Vector containers
        # Using a fixed reference for positioning
        v_group = VGroup()
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#00FF00")
        v = Arrow(start=ORIGIN, end=RIGHT*2, color=WHITE)
        label_v = MathTex(r"\vec{v}").scale(0.7)
        label_v.next_to(v.get_end(), UP, buff=0.1)
        
        container1 = VGroup(v, label_v, icon.copy())
        self.place_at_grid(v, "C5", scale_factor=0.7)
        # Re-attach label after scale
        label_v.next_to(v.get_end(), UP, buff=0.1)
        self.add(v, label_v)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color("#00FF00")
        
        v2 = Arrow(start=ORIGIN, end=RIGHT*4, color="#00FF00")
        label_v2 = MathTex(r"2\vec{v}").scale(0.7)
        self.place_at_grid(label_v2, "D5", scale_factor=0.6)
        
        self.play(Transform(v, v2), Transform(label_v, label_v2))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color("#FF0000")
        
        v_neg = Arrow(start=ORIGIN, end=LEFT*1, color="#FF0000")
        label_v_neg = MathTex(r"-0.5\vec{v}").scale(0.7)
        
        # Container as requested for final step
        anim_group = VGroup(v_neg, label_v_neg, icon.copy())
        self.place_in_area(anim_group, "B4", "E6", scale_factor=0.8)
        
        self.play(Transform(v, v_neg), Transform(label_v, label_v_neg))
        self.wait(2)
