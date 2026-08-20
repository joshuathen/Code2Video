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
        self.setup_layout("Application: The Archer Fish", [
            "The archer fish hunts insects above.",
            "Refraction shifts the bug’s apparent position.",
            "The fish adjusts its aim to strike."
        ])
        
        water_surface = Line(start=self.grid["C1"] + LEFT*1, end=self.grid["C6"] + RIGHT*1, color=BLUE)
        water_text = Text("Water", font_size=16, color=BLUE).next_to(water_surface, DOWN)
        
        # [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/fish.svg]
        archer_fish = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/fish.svg")
        self.place_at_grid(archer_fish, "E4", scale_factor=0.9)
        fish_label = Text("Archer Fish", font_size=16, color="#00BFFF").next_to(archer_fish, DOWN)
        self.add(fish_label)
        
        # [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/insect.svg]
        bug = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/insect.svg")
        self.place_at_grid(bug, "A5", scale_factor=0.8)
        bug_label = Text("Bug (True)", font_size=16, color=WHITE).next_to(bug, UP)
        self.add(bug_label)
        
        apparent_bug = Dot(color="#FFD700", radius=0.1)
        self.place_at_grid(apparent_bug, "A3", scale_factor=0.8)
        apparent_label = Text("Bug (Apparent)", font_size=16, color="#FFD700").next_to(apparent_bug, UP)
        self.add(apparent_label)
        
        # Light ray
        ray1 = Line(bug.get_center(), self.grid["C4"], color=WHITE)
        ray2 = Line(self.grid["C4"], archer_fish.get_center(), color=WHITE)
        illusion_ray = DashedLine(apparent_bug.get_center(), self.grid["C4"], color="#FFD700")

        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(water_surface, water_text), FadeIn(archer_fish), FadeIn(fish_label))
        self.lecture[0].set_color("#00BFFF")

        # === Animation for Lecture Line 2 ===
        self.play(FadeIn(bug, bug_label), FadeIn(apparent_bug, apparent_label), Create(ray1), Create(ray2), Create(illusion_ray))
        self.lecture[1].set_color("#FFD700")

        # === Animation for Lecture Line 3 ===
        aim_line = Line(archer_fish.get_center(), self.grid["C2"], color=RED, stroke_width=2)
        self.play(Create(aim_line))
        self.lecture[2].set_color(RED)
        
        self.wait(2)
