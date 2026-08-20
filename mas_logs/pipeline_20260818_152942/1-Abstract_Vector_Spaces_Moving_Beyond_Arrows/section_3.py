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
        lecture_lines = ["The eight axioms define a vector space.", "Test each axiom to verify the space.", "One failed axiom rejects the set."]
        self.setup_layout("Testing the Rules: The Axiom Checklist", lecture_lines)
        
        # Assets
        checklist = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/checklist.svg")
        marker = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/marker.svg")
        
        axioms = [
            "1. Closure (+)", "2. Commutativity", "3. Associativity", 
            "4. Identity", "5. Inverse", "6. Closure (sc)",
            "7. Distributivity 1", "8. Distributivity 2"
        ]
        
        axiom_list = VGroup(*[Text(a, font_size=16, color=WHITE) for a in axioms])
        axiom_list.arrange(DOWN, aligned_edge=LEFT)
        # Line 58 fix
        self.place_in_area(axiom_list, 'A3', 'F5', scale_factor=0.6)
        
        checklist_icon = checklist.copy().scale(0.3).next_to(axiom_list, LEFT)
        
        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(axiom_list), FadeIn(checklist_icon))
        self.lecture[0].set_color("#FFFFFF")
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#00FF00")
        markers = VGroup()
        for axiom in axiom_list:
            m = marker.copy().scale(0.1).next_to(axiom, LEFT, buff=0.1)
            markers.add(m)
            self.play(FadeIn(m), axiom.animate.set_color("#00FF00"), run_time=0.1)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FF0000")
        self.play(axiom_list[4].animate.set_color("#FF0000"))
        
        cross = Cross(axiom_list[4], color=RED).scale(0.5)
        self.play(Create(cross))
        self.wait(1)
        
        # Reset and Final highlight
        self.play(FadeOut(markers), FadeOut(cross), axiom_list.animate.set_color(WHITE))
        checklist_final = checklist.copy().scale(0.3).next_to(axiom_list, LEFT)
        self.play(FadeIn(checklist_final), axiom_list.animate.set_color("#00FFFF"))
        self.wait(1)
