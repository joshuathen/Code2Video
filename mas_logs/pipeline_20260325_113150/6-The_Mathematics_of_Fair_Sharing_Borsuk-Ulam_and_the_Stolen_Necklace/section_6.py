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
        # Mandatory call to setup_layout
        self.setup_layout("Conclusion and Visual Recap", [
            "Topology ensures fairness through continuous geometric shapes.",
            "Math proves the solution exists before we find it.",
            "The thieves can now share their loot perfectly."
        ])

        # === Animation for Lecture Line 1 ===
        # Highlight Line 1
        self.lecture[0].set_color(YELLOW)
        
        # Load assets
        ruby_path = "/mmfs1/data/home/jthen/Code2Video/assets/icon/ruby.svg"
        emerald_path = "/mmfs1/data/home/jthen/Code2Video/assets/icon/emerald.svg"
        
        # Create identical piles of beads
        def create_pile():
            r1 = SVGMobject(ruby_path)
            r2 = SVGMobject(ruby_path)
            e1 = SVGMobject(emerald_path)
            e2 = SVGMobject(emerald_path)
            pile = VGroup(r1, e1, r2, e2).arrange_in_grid(2, 2, buff=0.2)
            pile.scale(0.5)
            return pile
            
        pile_a = create_pile()
        pile_b = create_pile()
        
        # Labels for thieves
        label_a = Text("Thief A", font_size=24, color=WHITE)
        label_b = Text("Thief B", font_size=24, color=WHITE)
        
        # Placement using visual anchor system
        self.place_in_area(pile_a, "B1", "C3")
        self.place_in_area(pile_b, "B4", "C6")
        self.place_at_grid(label_a, "A2")
        self.place_at_grid(label_b, "A5")
        
        self.play(
            FadeIn(pile_a), FadeIn(pile_b),
            Write(label_a), Write(label_b)
        )
        self.wait(2)

        # === Animation for Lecture Line 2 ===
        # Reset previous highlight and highlight Line 2 with specific color
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color("#00FFFF")
        
        # TOPOLOGY with glow effect
        topology_text = Text("TOPOLOGY", font_size=72, color="#00FFFF")
        # Add a simple glow by stroke expansion
        glow = topology_text.copy().set_stroke(width=15, opacity=0.3, color="#00FFFF")
        topo_group = VGroup(glow, topology_text)
        
        # Place in a central area on the right grid
        self.place_in_area(topo_group, "D1", "F6")
        
        self.play(Write(topo_group))
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        # Reset previous highlight and highlight Line 3 with specific color
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color("#00FF00")
        
        # Final fade to Thieves A & B with Fair Split Checkmark
        self.play(
            FadeOut(pile_a), FadeOut(pile_b),
            FadeOut(label_a), FadeOut(label_b),
            FadeOut(topo_group)
        )
        
        thief_a_final = Text("A", font_size=48, color=WHITE)
        thief_b_final = Text("B", font_size=48, color=WHITE)
        self.place_at_grid(thief_a_final, "B2")
        self.place_at_grid(thief_b_final, "B5")
        
        # Checkmark and 'Fair Split' label in green
        checkmark = Text("✔", color="#00FF00", font_size=100)
        self.place_in_area(checkmark, "D3", "E4")
        
        fair_split_label = Text("Fair Split", color="#00FF00", font_size=32)
        self.place_in_area(fair_split_label, "F3", "F4")
        
        self.play(
            FadeIn(thief_a_final), FadeIn(thief_b_final),
            Write(checkmark), Write(fair_split_label)
        )
        self.wait(3)
