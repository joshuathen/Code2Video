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

class Section2Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Prerequisite Physics: Conservation Laws", 
                          ["Elastic collisions obey strict conservation laws.", 
                           "Momentum remains constant during every impact.", 
                           "Kinetic energy is perfectly preserved throughout."])
        
        # Setup visual elements
        # SVG asset integration
        billiard_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/billiard.svg")
        ball1 = Circle(radius=0.3, color=BLUE, fill_opacity=0.8)
        ball2 = Circle(radius=0.3, color=RED, fill_opacity=0.8)
        
        # === Animation for Lecture Line 1 ===
        # Display two balanced scales (using simple geometric representation for scales)
        scale_base = Rectangle(width=2.5, height=0.2, color="#00FF00")
        scale_rod = Line(start=UP*0.5, end=DOWN*0.5, color="#00FF00")
        scale_group = VGroup(scale_base, scale_rod)
        self.place_in_area(scale_group, 'C2', 'C5', scale_factor=1.0)
        
        self.play(Create(scale_group))
        self.lecture[0].set_color("#00FF00")
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Show energy flowing between two linked systems (using vectors)
        v1 = Vector(RIGHT*1.0, color="#33A1FF")
        v2 = Vector(LEFT*1.0, color="#33A1FF")
        
        self.place_at_grid(ball1, 'D2', scale_factor=0.8)
        self.place_at_grid(ball2, 'D5', scale_factor=0.8)
        self.place_at_grid(billiard_icon, 'D3', scale_factor=0.5)
        
        # Position vectors near the billiard icon
        v1.next_to(billiard_icon, LEFT, buff=0.1)
        v2.next_to(billiard_icon, RIGHT, buff=0.1)
        
        self.add(ball1, ball2, billiard_icon)
        self.play(FadeIn(v1), FadeIn(v2))
        self.lecture[1].set_color("#FF00FF") # Momentum color
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Animate the total energy remaining constant
        energy_label = Text("Total E = Const", font_size=24, color="#FFFF00")
        self.place_in_area(energy_label, 'B3', 'B4', scale_factor=0.9)
        
        self.play(Write(energy_label))
        self.lecture[2].set_color("#FFFF00")
        self.wait(2)
