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

class Section5Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Conclusion & Real-World Application", ["Vectors build our digital world.", "Linear algebra enables game physics.", "Machine learning uses vector spaces."])
        
        # === Animation for Lecture Line 1 ===
        # Show an icon representing a physical force vector using [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/controller.svg]. Color in #FFFFFF.
        icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/controller.svg", color="#FFFFFF")
        label1 = Text("Force Vector", font_size=20, color="#FFFFFF")
        group1 = VGroup(icon, label1).arrange(DOWN)
        self.place_in_area(group1, 'A4', 'C6', scale_factor=0.6)
        self.play(FadeIn(icon), Write(label1))
        self.lecture[0].set_color("#FFFFFF")
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Animate vectors combining to represent velocity in a game. Color in #FF5733.
        v1 = Arrow(ORIGIN, UP*0.8 + RIGHT*0.8, color="#FF5733", buff=0)
        v2 = Arrow(ORIGIN, DOWN*0.4 + RIGHT*1.2, color="#FF5733", buff=0)
        v_sum = Arrow(ORIGIN, UP*0.4 + RIGHT*2.0, color="#FF5733", buff=0)
        resultant_label = Text("Resultant Velocity", font_size=20, color="#FF5733")
        
        group2 = VGroup(v1, v2, v_sum)
        self.place_in_area(group2, 'D4', 'E6', scale_factor=0.7)
        self.place_at_grid(resultant_label, 'D3', scale_factor=0.7)
        
        self.play(Create(v1), Create(v2))
        self.play(ReplacementTransform(VGroup(v1.copy(), v2.copy()), v_sum))
        self.play(Write(resultant_label))
        self.lecture[1].set_color("#FF5733")
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Display text: 'Vectors are essential for physics and graphics' alongside [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/computer.svg]. Color in #2ECC71.
        msg = Text("Vectors are essential for physics and graphics", font_size=20, color="#2ECC71")
        icon2 = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/computer.svg", color="#2ECC71")
        group3 = VGroup(icon2, msg).arrange(RIGHT)
        
        self.place_in_area(group3, 'F1', 'F6', scale_factor=0.65)
        self.play(FadeIn(group3))
        self.lecture[2].set_color("#2ECC71")
        self.wait(2)
