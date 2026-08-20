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
        lecture_lines = [
            "Facts aren't single nodes, but constellations.",
            "Activation lights up a specific neural path.",
            "This forms the knowledge representation pattern.",
            "Spotlight effect shows fact distribution clearly.",
            "Connections define the stored relational truth."
        ]
        self.setup_layout("Visualizing Weights as Knowledge", lecture_lines)
        self.lecture.set_opacity(0)
        
        # Setup visual elements
        vec1 = Vector([1, 0.5], color="#E0FFFF")
        vec2 = Vector([-0.5, 1], color="#E0FFFF")
        group = VGroup(vec1, vec2)
        
        # Using area for constellations (B004, B035, B019)
        self.place_in_area(group, 'C4', 'E6', scale_factor=0.8)
        
        label = Text("Learned Knowledge", font_size=20, color=WHITE).scale(0.7) # B020
        label.next_to(group, UP, buff=0.1) # B011
        
        spotlight = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/spotlight.svg")
        self.place_at_grid(spotlight, 'B5', scale_factor=0.5)
        spotlight.set_opacity(0)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_opacity(1)
        self.play(Create(group), Write(label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_opacity(1)
        self.play(group.animate.set_color("#FFD700"))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_opacity(1)
        self.play(Indicate(group, color="#FFD700"))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_opacity(1)
        self.play(FadeIn(spotlight))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_opacity(1)
        self.play(FadeOut(group), FadeOut(label), FadeOut(spotlight))
        self.wait(1)
