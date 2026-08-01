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
        # Setup the layout with section title and lecture lines
        # Titles and lecture lines are based on the fetched storyboard
        self.setup_layout("The Birth of the Dual Graph", [
            "Every planar graph has a unique dual companion.",
            "Duality transforms the original structure into a new graph.",
            "Place a dual vertex inside every original face."
        ])
        
        # --- PREPARE ASSETS ---
        # House Graph Asset [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/house.svg]
        # Using SVGMobject as required by issue 33.
        house_graph = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/house.svg", color=WHITE)
        self.place_in_area(house_graph, "A2", "D4", scale_factor=2.0)

        # Dual Vertices (V*)
        # dv1: Inside square face. Issue 19: use D3.
        dv1 = Dot(color="#FF0000")
        self.place_at_grid(dv1, "D3", scale_factor=1.2) 
        l1 = Text("V*", font_size=16, color=WHITE).next_to(dv1, RIGHT, buff=0.1)

        # dv2: Inside triangle face.
        dv2 = Dot(color="#FF0000")
        self.place_at_grid(dv2, "B3", scale_factor=1.2) 
        l2 = Text("V*", font_size=16, color=WHITE).next_to(dv2, RIGHT, buff=0.1)

        # dv3: External face. Issue 20: use E5.
        dv3 = Dot(color="#FF0000")
        self.place_at_grid(dv3, "E5", scale_factor=1.2) 
        l3 = Text("V*", font_size=16, color=WHITE).next_to(dv3, RIGHT, buff=0.1)

        dual_elements = VGroup(dv1, l1, dv2, l2, dv3, l3)

        # === Animation for Lecture Line 1 ===
        # "Every planar graph has a unique dual companion."
        self.lecture[0].set_color(WHITE)
        self.play(DrawBorderThenFill(house_graph), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # "Duality transforms the original structure into a new graph."
        # Visual: Dim the house graph to light gray (#D3D3D3) as per storyboard.
        self.lecture[1].set_color("#D3D3D3")
        self.play(house_graph.animate.set_color("#D3D3D3"), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # "Place a dual vertex inside every original face."
        # Visual: Fade in the 3 red dots with V* labels.
        self.lecture[2].set_color("#FF0000")
        self.play(FadeIn(dual_elements), run_time=2)
        self.wait(2)
