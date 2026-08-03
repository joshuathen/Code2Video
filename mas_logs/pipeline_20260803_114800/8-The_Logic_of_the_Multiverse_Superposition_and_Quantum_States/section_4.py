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

class Section4Scene(TeachingScene):
    def construct(self):
        self.setup_layout(
            "The Measurement Problem: Wavefunction Collapse", 
            [
                "Looking at a quantum system forces a definite choice.",
                "Superposition collapses into a single classical outcome.",
                "Probability determines which state the system settles into."
            ]
        )
        
        # Assets
        box_asset_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/box.svg"
        cat_asset_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/cat.png"

        # === Animation for Lecture Line 1 ===
        # Show a brown box [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/box.svg] (#A52A2A) 
        # labeled 'Schrödinger’s Box' in the center.
        # Display two faint, transparent cat icons [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/cat.png] (#FFFFFF) 
        # inside the box.
        
        self.lecture[0].set_color(YELLOW)
        
        box = SVGMobject(box_asset_path).set_color("#A52A2A")
        self.place_in_area(box, "B2", "E5", scale_factor=1.8)
        
        box_label = Text("Schrödinger's Box", font_size=20, color="#A52A2A")
        box_label.next_to(box, DOWN, buff=0.1)
        
        # Mystery symbol
        q_mark = Text("?", font_size=50, color=WHITE, fill_opacity=0.8)
        self.place_in_area(q_mark, "C3", "D4")
        
        # Cats inside (faint)
        cat_alive = ImageMobject(cat_asset_path).set_opacity(0.25)
        cat_dead = ImageMobject(cat_asset_path).set_opacity(0.25)
        
        # Position cats using grid
        self.place_at_grid(cat_alive, "C3", scale_factor=0.6)
        self.place_at_grid(cat_dead, "D4", scale_factor=0.6)
        
        self.play(FadeIn(box), FadeIn(box_label), Write(q_mark))
        self.play(FadeIn(cat_alive), FadeIn(cat_dead))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Animate the box [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/box.svg] opening 
        # and the '?' symbol disappearing.
        # Snap the cat [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/cat.png] 
        # into a single green 'Alive' state (#00FF00).
        # Fade out the other possible state [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/cat.png] 
        # to show wavefunction collapse.
        
        self.play(self.lecture[0].animate.set_color(WHITE))
        self.lecture[1].set_color(YELLOW)
        
        # Issue 26: Fix labels positions (C2, D5)
        alive_label = Text("Alive", font_size=18, color="#00FF00")
        dead_label = Text("Dead", font_size=18, color="#FF0000")
        self.place_at_grid(alive_label, 'C2', scale_factor=0.8)
        self.place_at_grid(dead_label, 'D5', scale_factor=0.8)

        # Opening animation: lift the box up and away
        self.play(
            box.animate.shift(UP * 1.5).set_opacity(0.5),
            box_label.animate.shift(UP * 1.5).set_opacity(0.5),
            FadeOut(q_mark),
            run_time=1.5
        )
        
        # Wavefunction Collapse
        # Use a green glow around the "alive" cat as ImageMobject doesn't support set_color
        glow = SurroundingRectangle(cat_alive, color="#00FF00", buff=0.1, fill_opacity=0.3)
        
        self.play(
            cat_alive.animate.set_opacity(1.0).scale(1.1),
            FadeIn(glow),
            FadeIn(alive_label),
            cat_dead.animate.set_opacity(0.05),
            FadeIn(dead_label, opacity=0.3),
            run_time=1.5
        )
        
        self.play(
            FadeOut(cat_dead),
            FadeOut(dead_label),
            run_time=0.8
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Probability determines which state the system settles into.
        
        self.play(self.lecture[1].animate.set_color(WHITE))
        self.lecture[2].set_color(YELLOW)
        
        # Issue 27: Position prob_note in area F1-F6
        prob_note = Text("Probability Collapse: 1.0", font_size=22, color=YELLOW)
        self.place_in_area(prob_note, 'F1', 'F6', scale_factor=0.7)
        
        result_rect = SurroundingRectangle(glow, color="#00FF00", buff=0.2)
        
        self.play(
            Create(result_rect),
            Write(prob_note)
        )
        self.play(
            result_rect.animate.scale(1.1).set_stroke(width=6),
            rate_func=there_and_back,
            run_time=2
        )
        self.wait(2)
        
        # Final cleanup
        self.play(
            *[FadeOut(m) for m in self.mobjects if m != self.title and m != self.lecture]
        )
        self.play(self.lecture[2].animate.set_color(WHITE))
        self.wait(1)
